# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import bisect
from dataclasses import dataclass, field
from typing import cast, final, Dict, Optional, Set

import numpy as np
from lucupy.minimodel import NightIndex, ObservationClass, Site, TimeslotIndex, ObservationStatus, TooType, Band

from scheduler.core.components.base import SchedulerComponent
from scheduler.core.components.collector import Collector
from scheduler.core.components.selector import Selector
from scheduler.core.eventsqueue import (Event, EveningTwilightEvent, InterruptionEvent, InterruptionResolutionEvent,
                                        MorningTwilightEvent, WeatherChangeEvent, ToOActivationEvent)
from scheduler.core.plans import Plans, Visit, Plan
from scheduler.services.logger_factory import create_logger
from .time_coordinate_record import TimeCoordinateRecord


__all__ = [
    'ChangeMonitor',
]


_logger = create_logger(__name__)


@final
@dataclass
class ChangeMonitor(SchedulerComponent):
    collector: Collector
    selector: Selector

    # A set of InterruptionEvents that prevent a site from being active.
    # They must be resolved by InterruptionResolutionEvents to reactivate the site.
    _blocking_event_sets: Dict[Site, Set[InterruptionEvent]] = field(init=False, default_factory=dict)

    def __post_init__(self):
        """
        Create the blocking event set for each site.
        """
        self._blocking_event_sets = {site: set() for site in self.collector.sites}

    def _process_blocking_event(self, event: InterruptionEvent) -> None:
        """
        Event handling routine where a blocking event is received and recorded for a site.
        """
        blocking_event_set = self._blocking_event_sets[event.site]
        if any(event.id == evt.id for evt in blocking_event_set):
            raise RuntimeError(f'Blocking event {event} for site {event.site} is already in the blocking list.')
        blocking_event_set.add(event)

    def _process_blocking_resolution_event(self, event: InterruptionResolutionEvent) -> None:
        """
        Event handling routine where a resolution event is processed to try to resolve a blocking event at a site.
        """
        blocking_event_set = self._blocking_event_sets[event.site]
        resolved_event: Optional[InterruptionEvent] = None
        for evt in blocking_event_set:
            if event.uuid_referenced == evt.id:
                resolved_event = evt
                break

        if resolved_event is None:
            raise RuntimeError(f'Resolution event {event} for site {event.site} does not resolve anything in the '
                               'blocking list.')

        blocking_event_set.remove(resolved_event)

    def is_site_unblocked(self, site: Site) -> bool:
        """
        Return True if the given site is NOT blocked by one or more events that need to be resolved before the site
        can have a plan, and False if the site is blocked.
        """
        return len(self._blocking_event_sets[site]) == 0

    def is_site_blocked(self, site: Site) -> bool:
        return len(self._blocking_event_sets[site]) > 0

    @staticmethod
    def is_interrupting_visit(plan: Plan, event_timeslot: TimeslotIndex) -> Optional[Visit]:
        # Check if there is a visit in the middle of the event and return such Visit if not return None.
        # Sort the visits by start time and find the one (if any) that happens just before this event.
        sorted_visits = sorted(plan.visits, key=lambda v: v.start_time_slot)
        visit_idx = bisect.bisect_right([v.start_time_slot for v in sorted_visits], event_timeslot) - 1
        visit = None if visit_idx < 0 else sorted_visits[visit_idx]
        return visit


    def process_event(self,
                      site: Site,
                      event: Event,
                      plans: Optional[Plans],
                      night_idx: NightIndex) -> Optional[TimeCoordinateRecord]:
        """
        TODO: Might want to make return type a Tuple[NightIndex, TimeslotIndex].

        Given an event occurring at a given site and an optional plan running on the site, determine
        the next timeslot where the plan should be recalculated, if any.

        The plan is optional, because for EveningTwilightEvent, no plan has yet been computed and thus None should
        be passed in.

        If the ChangeMonitor determines that a new plan should be calculated, then a timeslot index indicating
        when the plan should be calculated is returned, and if no new plan should be calculated, then None is returned.

        :param site: the site at which the event occurred
        :param event: the event that occurred
        :param plans: the plans that are currently in action (if any) for the night, which consist of a plan per site
        :param night_idx: the night index

        :return: a time coordinate record which provides information about when the next plan should be computed
                 (if any), if a night is done, and if time accounting should be performed
        """
        # Convert the site to a name here in case we want to use site.name (key name) or site.site_name (long form).
        site_name = site.name

        # Translate the local event time to Scheduler time coordinates.
        night_events = self.collector.get_night_events(site)
        twi_eve = night_events.twilight_evening_12[night_idx].to_datetime(site.timezone)
        event_timeslot = event.to_timeslot_idx(twi_eve, self.collector.time_slot_length.to_datetime())
        num_timeslots_for_night = night_events.num_timeslots_per_night[night_idx]
        last_timeslot_for_night = TimeslotIndex(num_timeslots_for_night - 1)

        # Process the event based on its type:
        match event:
            case EveningTwilightEvent():
                # We always create a plan for the evening twilight event.
                if event_timeslot != 0:
                    _logger.warning(f'EveningTwilightEvent for site {site_name} should be scheduled for timeslot 0, '
                                    f'but is scheduled for timeslot {event_timeslot}.')
                # We do not perform any time accounting for the evening twilight.
                return TimeCoordinateRecord(event=event,
                                            timeslot_idx=event_timeslot,
                                            perform_time_accounting=False)

            case MorningTwilightEvent():
                if event_timeslot != last_timeslot_for_night:
                    _logger.warning(f'MorningTwilightEvent for site {site_name} should be scheduled for '
                                    f'timeslot {last_timeslot_for_night} but is scheduled for '
                                    f'timeslot {event_timeslot}.')
                return TimeCoordinateRecord(event=event,
                                            timeslot_idx=last_timeslot_for_night,
                                            done=True)

            case WeatherChangeEvent(variant_change=variant_change):
                # Regardless, we want to change the weather values for CC and IQ.
                self.selector.update_site_variant(site, variant_change)

                # If the site is blocked, we have no reason to recalculate a plan until all blocking events
                # are unblocked.
                if plans is None:
                    if self.is_site_blocked(site):
                        return None
                    raise ValueError(f'No plans have been created for night {night_idx}.')

                # Check if there is a visit running now.
                plan = plans[site]
                if plan is None:
                    return TimeCoordinateRecord(event=event,
                                                timeslot_idx=event_timeslot)

                visit = self.is_interrupting_visit(plan, event_timeslot)

                # If there are no visits in progress, then recalculate now.
                if visit is None:
                    return TimeCoordinateRecord(event=event,
                                                timeslot_idx=event_timeslot)

                # Check if event occurs after the calculated visit is already over. If so, recalculate now.
                visit_end_time_slot = visit.start_time_slot + visit.time_slots - 1
                if visit_end_time_slot < event_timeslot:
                    return TimeCoordinateRecord(event=event,
                                                timeslot_idx=event_timeslot)

                # Otherwise, we are in the middle of a visit and interrupt it.
                remaining_time_slots = visit_end_time_slot - event_timeslot + 1

                # TODO: This should be more complicated to allow for splitting and to meet requirements.
                # TODO: Talk to Bryan about how to go about this.
                obs = Collector.get_observation(visit.obs_id)

                # Most restrictive conditions.
                mrc = obs.constraints.conditions

                # TODO: This code is somewhat duplicated from Selector. See if we can simplify it, although in this
                # TODO: case, it is for a single night instead of all nights.
                target_info = Collector.get_target_info(obs.id)
                if obs.obs_class in [ObservationClass.SCIENCE, ObservationClass.PROGCAL]:
                    neg_ha = target_info[night_idx].hourangle[0].value < 0
                else:
                    neg_ha = False
                too_type = obs.too_type

                # Create a variant representing the weather values and match conditions on it to see if we can
                # complete the executing visit.
                # The values here will all be 0 or 1 since the Variant has consistent values, but check regardless if
                # there is a timeslot where we cannot execute the visit. If so, recalculate the plan now.
                variant = variant_change.make_variant(remaining_time_slots)
                slot_values = Selector.match_conditions(mrc, variant, neg_ha, too_type)
                if not np.all(slot_values > 0):
                    return TimeCoordinateRecord(event=event,
                                                timeslot_idx=event_timeslot)

                # Otherwise, we can finish the observation. Start the weather change at time slot after the visit ends.
                return TimeCoordinateRecord(event=event,
                                            timeslot_idx=TimeslotIndex(visit_end_time_slot + 1))

            case InterruptionEvent():
                self._process_blocking_event(cast(InterruptionEvent, event))
                return TimeCoordinateRecord(event=event,
                                            timeslot_idx=event_timeslot)

            case InterruptionResolutionEvent():
                # There is no plan if there is a block, so we do not perform time accounting in this case.
                self._process_blocking_resolution_event(cast(InterruptionResolutionEvent, event))
                return TimeCoordinateRecord(event=event,
                                            timeslot_idx=event_timeslot,
                                            perform_time_accounting=False)

            case ToOActivationEvent(too_id=too_id):

                # If the site is blocked, we have no reason to recalculate a plan until all blocking events
                # are unblocked.
                if plans is None:
                    if self.is_site_blocked(site):
                        return None
                    raise ValueError(f'No plans have been created for night {night_idx}.')

                # Check if there is a visit running now. If there is not anything
                plan = plans[site]
                if plan is None:
                    return TimeCoordinateRecord(event=event,
                                                timeslot_idx=event_timeslot)

                visit = self.is_interrupting_visit(plan, event_timeslot)

                # If there are no visits in progress, then recalculate now.
                if visit is None:
                    return TimeCoordinateRecord(event=event,
                                                timeslot_idx=event_timeslot)

                # Check if event occurs after the calculated visit is already over. If so, recalculate now.
                visit_end_time_slot = visit.start_time_slot + visit.time_slots - 1
                if visit_end_time_slot < event_timeslot:
                    return TimeCoordinateRecord(event=event,
                                                timeslot_idx=event_timeslot)

                # Otherwise, we are in the middle of a visit and interrupt it.
                remaining_time_slots = visit_end_time_slot - event_timeslot + 1

                past_obs = self.collector.get_observation(visit.obs_id)
                program = self.collector.get_program(past_obs.belongs_to)

                # If the visit we are interrupting is Band4, interrupt the visit.
                if program.band is Band.BAND4:
                    return TimeCoordinateRecord(event=event,
                                                timeslot_idx=event_timeslot)

                # Otherwise put everything after the visit and do not interrupt.
                too = self.collector.get_observation(too_id)

                # Check that only Rapid ToOs are updated
                if too.too_type is not TooType.RAPID:
                    raise ValueError(f'ToO {too_id} is not RAPID.')

                # Change status of an observation
                too.status = ObservationStatus.READY
                return TimeCoordinateRecord(event=event,
                                            timeslot_idx=TimeslotIndex(event_timeslot + 1))

            # For now, for all other events, just recalculate immediately.
            case _:
                return TimeCoordinateRecord(event=event,
                                            timeslot_idx=event_timeslot)
