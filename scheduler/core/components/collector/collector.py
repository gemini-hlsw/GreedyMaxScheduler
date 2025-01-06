# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import time
from dataclasses import dataclass
from inspect import isclass
from typing import ClassVar, Dict, FrozenSet, Iterable, List, Optional, Tuple, Type, final

import astropy.units as u
import numpy as np

from astropy.time import Time, TimeDelta

from lucupy.minimodel import (ALL_SITES, NightIndex, NightIndices,
                              Observation, ObservationID, ObservationClass, Program, ProgramID, ProgramTypes, Semester,
                              Site, Target, TimeslotIndex, QAState, ObservationStatus, SiderealTarget, NonsiderealTarget,
                              Group, SkyBackground, ElevationType, Constraints)
from lucupy.timeutils import time2slots
from lucupy.types import Day, ZeroTime
from lucupy import sky

from scheduler.core.calculations.nightevents import NightEvents
from scheduler.core.calculations.targetinfo import TargetInfo, TargetInfoMap, TargetInfoNightIndexMap
from scheduler.core.components.base import SchedulerComponent
from scheduler.core.components.nighteventsmanager import NightEventsManager
from scheduler.core.plans import Plans, Visit
from scheduler.core.programprovider.abstract import ProgramProvider
from scheduler.core.sources.sources import Sources
from scheduler.services import logger_factory
from scheduler.services.ephemeris import EphemerisCalculator
from scheduler.services.proper_motion import ProperMotionCalculator
from scheduler.services.resource import NightConfiguration
from scheduler.services.resource import ResourceService
from scheduler.services.visibility.calculator import calculate_target_snapshot, visibility_calculator

__all__ = [
    'Collector',
]

logger = logger_factory.create_logger(__name__)


# TODO: Merge this if possible with Visit.
# TODO: This is just used internally to the Collector and thus we do not export it outside of this package.
@final
@dataclass(frozen=True)
class GroupVisits:
    """Container for holding group information for each visit"""
    group: Group
    visits: List[Visit]

    def start_time_slot(self):
        if not self.visits:
            raise RuntimeError(f'start_time_slot requested, but no visits recorded for {self.group.unique_id}')
        # return min([v.start_time_slot for v in self.visits])
        return self.visits[0].start_time_slot

    def end_time_slot(self):
        if not self.visits:
            raise RuntimeError(f'end_time_slot requested, but no visits recorder for {self.group.unique_id}')
        return self.visits[-1].start_time_slot + self.visits[-1].time_slots - 1


@final
@dataclass
class Collector(SchedulerComponent):
    """
    The interval [start_vis_time, end_vis_time] indicates the time interval that we want to consider during
    the scheduling for visibility time. Note that the generation of plans will begin on the night indicated by
    start_vis_time and proceed for num_nights_to_schedule, a parameter passed to the Selector, which must
    represent fewer nights than in the [start_vis_time, end_vis_time] schedule.

    Also note that we never have need to calculate visibility retroactively, hence why plan generation begins
    on the night of start_vis_time.

    Here, we just perform the necessary calculations, and are not concerned with the number of nights to be
    scheduled.
    """
    start_vis_time: Time
    end_vis_time: Time
    num_of_nights: int
    sites: FrozenSet[Site]
    semesters: FrozenSet[Semester]
    sources: Sources
    with_redis: bool
    time_slot_length: TimeDelta
    program_types: FrozenSet[ProgramTypes]
    obs_classes: FrozenSet[ObservationClass]

    # Manage the NightEvents with a NightEventsManager to avoid unnecessary recalculations.
    _night_events_manager: ClassVar[NightEventsManager] = NightEventsManager()

    # Resource service.
    # TODO: This will be moved out when event processing is handled.
    _resource_service: ClassVar[ResourceService]

    # This should not be populated, but we put it here instead of in __post_init__ to eliminate warnings.
    # This is a list of the programs as read in.
    # We only want to read these in once unless the program_types change, which they should not.
    _programs: ClassVar[Dict[ProgramID, Program]] = {}

    # A set of ObservationIDs per ProgramID.
    _observations_per_program: ClassVar[Dict[ProgramID, FrozenSet[ObservationID]]] = {}

    # This is a map of observation information that is computed as the programs
    # are read in. It contains both the Observation and the base Target (if any) for
    # the observation.
    _observations: ClassVar[Dict[ObservationID, Tuple[Observation, Optional[Target]]]] = {}

    # The target information is dependent on the:
    # 1. TargetName
    # 2. ObservationID (for the associated constraints and site)
    # 4. NightIndex of interest
    # We want the ObservationID in here so that any target sharing in GPP is deliberately split here, since
    # the target info is observation-specific due to the constraints and site.
    _target_info: ClassVar[TargetInfoMap] = {}

    # The default timeslot length currently used.
    DEFAULT_TIMESLOT_LENGTH: ClassVar[Time] = 1.0 * u.min

    # These are exclusive to the create_time_array.
    _MIN_NIGHT_EVENT_TIME: ClassVar[Time] = Time('1980-01-01 00:00:00', format='iso', scale='utc')

    # NOTE: This logs an ErfaWarning about dubious year. This is due to using a future date and not knowing
    # how many leap seconds have happened: https://github.com/astropy/astropy/issues/5809
    _MAX_NIGHT_EVENT_TIME: ClassVar[Time] = Time('2100-01-01 00:00:00', format='iso', scale='utc')

    def __post_init__(self):
        """
        Initializes the internal data structures for the Collector and populates them.
        """
        # Check that the times are valid.
        if not np.isscalar(self.start_vis_time.value):
            msg = f'Illegal start time (must be scalar): {self.start_vis_time}.'
            raise ValueError(msg)
        if not np.isscalar(self.end_vis_time.value):
            msg = f'Illegal end time (must be scalar): {self.end_vis_time}.'
            raise ValueError(msg)
        if self.start_vis_time > self.end_vis_time:
            msg = f'Start time ({self.start_vis_time}) cannot occur later than end time ({self.end_vis_time}).'
            raise ValueError(msg)

        # Set up the time grid for the period under consideration in calculations: this is an astropy Time
        # object from start_time to end_time inclusive, with one entry per day.
        # Note that the format is in jdate.
        self.time_grid = Time(np.arange(self.start_vis_time.jd,
                                        self.end_vis_time.jd + 1.0, (1.0 * u.day).value),
                              format='jd')

        # The number of nights for which we are performing calculations.
        self.num_nights_calculated = len(self.time_grid)

        # TODO: This code can be greatly simplified. The night_events only have to be calculated once.
        # Create the night events, which contain the data for all given nights by site.
        # This may retrigger a calculation of the night events for one or more sites.
        self.night_events = {
            site: Collector._night_events_manager.get_night_events(self.time_grid, self.time_slot_length, site)
            for site in self.sites
        }
        Collector._resource_service = self.sources.origin.resource

    def get_night_events(self, site: Site) -> NightEvents:
        return Collector._night_events_manager.get_night_events(self.time_grid,
                                                                self.time_slot_length,
                                                                site)

    @staticmethod
    def get_program_ids() -> Iterable[ProgramID]:
        """
        Return a list of all the program IDs stored in the Collector.
        """
        return Collector._programs.keys()

    @staticmethod
    def get_program(program_id: ProgramID) -> Optional[Program]:
        """
        If a program with the given ID exists, return it.
        Otherwise, return None.
        """
        return Collector._programs.get(program_id, None)

    @staticmethod
    def get_all_observations() -> Iterable[Observation]:
        return [obs_data[0] for obs_data in Collector._observations.values()]

    @staticmethod
    def get_observation_ids(program_id: Optional[ProgramID] = None) -> Optional[Iterable[ObservationID]]:
        """
        Return the observation IDs in the Collector.
        If the prog_id is specified, limit these to those in the specified in the program.
        If no such prog_id exists, return None.
        If no prog_id is specified, return a complete list of observation IDs.
        """
        if program_id is None:
            return Collector._observations.keys()
        return Collector._observations_per_program.get(program_id, None)

    @staticmethod
    def get_observation(obs_id: ObservationID) -> Optional[Observation]:
        """
        Given an ObservationID, if it exists, return the Observation.
        If not, return None.
        """
        value = Collector._observations.get(obs_id, None)
        return None if value is None else value[0]

    @staticmethod
    def get_base_target(obs_id: ObservationID) -> Optional[Target]:
        """
        Given an ObservationID, if it exists and has a base target, return the Target.
        If one of the conditions is not met, return None.
        """
        value = Collector._observations.get(obs_id, None)
        return None if value is None else value[1]

    @staticmethod
    def get_observation_and_base_target(obs_id: ObservationID) -> Optional[Tuple[Observation, Optional[Target]]]:
        """
        Given an ObservationID, if it exists, return the Observation and its Target.
        If not, return None.
        """
        return Collector._observations.get(obs_id, None)

    @staticmethod
    def get_target_info(obs_id: ObservationID) -> Optional[TargetInfoNightIndexMap]:
        """
        Given an ObservationID, if the observation exists and there is a target for the
        observation, return the target information as a map from NightIndex to TargetInfo.
        """
        info = Collector.get_observation_and_base_target(obs_id)
        if info is None:
            return None

        obs, target = info
        if target is None:
            return None

        target_name = target.name
        return Collector._target_info.get((target_name, obs_id), None)

    @staticmethod
    def _process_timing_windows(prog: Program, obs: Observation) -> List[Time]:
        """
        Given an Observation, convert the TimingWindow information in it to a simpler format
        to verify by converting each TimingWindow representation to a collection of Time frames
        based on the start, duration, repeat, and period.

        If no timing windows are given, then create one large timing window for the entire program.

        TODO: Look into simplifying to datetime instead of AstroPy Time.
        TODO: We may want to store this information in an Observation for future use.
        """
        if not obs.constraints or len(obs.constraints.timing_windows) == 0:
            # Create a timing window for the entirety of the program.
            windows = [Time([prog.start, prog.end])]
        else:
            windows = []
            for tw in obs.constraints.timing_windows:
                # The start time is now already an astropy Time
                begin = tw.start
                duration = tw.duration.total_seconds() / 3600.0 * u.h
                repeat = max(1, tw.repeat)
                period = tw.period.total_seconds() / 3600.0 * u.h if tw.period is not None else 0.0 * u.h
                windows.extend([Time([begin + i * period, begin + i * period + duration]) for i in range(repeat)])

        return windows

    def _calculate_target_info(self,
                               obs: Observation,
                               target: Target) -> TargetInfoNightIndexMap:
        """
        For a given site, calculate the information for a target for all the nights in
        the time grid and store this in the _target_information.

        Some of this information may be repetitive as, e.g. the RA and dec of a target should not
        depend on the site, so sites whose twilights overlap with have this information repeated.

        Finally, this method can calculate the total amount of time that, for the observation,
        the target is visible, and the visibility fraction for the target as a ratio of the amount of
        time remaining for the observation to the total visibility time for the target from a night through
        to the end of the period.
        """
        # Get the night events.
        if obs.site not in self.night_events:
            raise ValueError(f'Requested obs {obs.id.id} target info for site {obs.site}, which is not included.')
        night_events = self.night_events[obs.site]

        target_vis = visibility_calculator.get_target_visibility(obs, self.time_grid, self.semesters)

        target_info: TargetInfoNightIndexMap = {}

        for i in range(self.num_nights_calculated):
            night_idx = NightIndex(i)
            target_snapshot = calculate_target_snapshot(night_idx,
                                                        obs,
                                                        target,
                                                        night_events,
                                                        self.time_grid[night_idx],
                                                        self.time_slot_length)
            ts = target_vis[night_idx]

            ti = TargetInfo(coord=target_snapshot.coord,
                            alt=target_snapshot.alt,
                            az=target_snapshot.az,
                            par_ang=target_snapshot.par_ang,
                            hourangle=target_snapshot.hourangle,
                            airmass=target_snapshot.airmass,
                            sky_brightness=target_snapshot.sky_brightness,
                            visibility_slot_idx=ts.visibility_slot_idx,
                            visibility_time=ts.visibility_time,
                            rem_visibility_time=ts.rem_visibility_time,
                            rem_visibility_frac=ts.rem_visibility_frac)

            target_info[NightIndex(night_idx)] = ti
        # Return all the target info for the base target in the Observation across the nights of interest.
        return target_info

    def load_programs(self, program_provider_class: Type[ProgramProvider], data: Iterable[dict]) -> None:
        """
        Load the programs provided as JSON or GPP disctionaries into the Collector.

        The program_provider should be a concrete implementation of the API to read in
        programs.

        The json_data comprises the program inputs as an iterable object per site. We use iterable
        since the amount of data here might be enormous, and we do not want to store it all
        in memory at once.

        In an OCS Program, all observations are guaranteed to be at the same site;
        however, since this may not always be the case and will not in GPP, we still process all programs
        and simply omit observations that are not at a site listed in the desired sites.
        """
        if not (isclass(program_provider_class) and issubclass(program_provider_class, ProgramProvider)):
            raise ValueError('Collector load_programs requires a ProgramProvider class as the second argument')
        program_provider = program_provider_class(self.obs_classes, self.sources)

        # Purge the old programs and observations.
        Collector._programs = {}

        # Keep a list of the observations for parallel processing.
        parsed_observations: List[Tuple[ProgramID, Observation]] = []

        # Read in the programs.
        # Count the number of parse failures.
        bad_program_count = 0

        for next_program in data:
            try:
                if len(next_program.keys()) == 1:
                    # Extract the data from the OCS JSON program. We do not need the top label.
                    next_data = next(iter(next_program.values()))
                else:
                    # This is a dictionary from GPP
                    next_data = next_program
                program = program_provider.parse_program(next_data)

                # If program could not be parsed, skip. This happens in one of three cases:
                # 1. Program semester cannot be determined from ID.
                # 2. Program type cannot be determined from ID.
                # 3. Program root group is empty.
                if program is None:
                    continue

                # TODO: improve this. Pass the semesters into the program_provider and return None as soon
                # TODO: as we know that the program is not from a semester in which we are interested.
                # If program semester is not in the list of specified semesters, skip.
                if program.semester is None or program.semester not in self.semesters:
                    logger.debug(f'Program {program.id} has semester {program.semester} (not included, skipping).')
                    continue

                # If a program has no time awarded, then we will get a divide by zero in scoring, so skip it.
                if program.program_awarded() == ZeroTime:
                    logger.debug(f'Program {program.id} has awarded time of zero (skipping).')
                    continue

                # If a program ID is repeated, warn and overwrite.
                if program.id in Collector._programs.keys():
                    logger.warning(f'Data contains a repeated program with id {program.id} (overwriting).')

                Collector._programs[program.id] = program

                # Set the observation IDs for this program.
                # We only want the observations that are located at the sites supported by the collector.
                # TODO: In GPP, if an AndGroup exists where the observations are not all from the same site, then
                # TODO: this should be an error.
                # TODO: In the case of an OrGroup, we only want:
                # TODO: 1. The branches that are OrGroups and are nonempty (i.e. have obs).
                # TODO: 2. The branches that are AndGroups and are nonempty (i.e. all obs are from the same site).
                # TODO: Applying this logic recursively should ensure only Groups that can be completed are included.
                site_supported_obs = [obs for obs in program.observations() if obs.site in self.sites]
                if site_supported_obs:
                    Collector._observations_per_program[program.id] = frozenset(obs.id for obs in site_supported_obs)
                    parsed_observations.extend((program.id, obs) for obs in site_supported_obs)

            except Exception as e:
                bad_program_count += 1
                logger.warning(f'Could not parse program: {e}')

        if bad_program_count:
            logger.error(f'Could not parse {bad_program_count} programs.')

        # TODO STEP 1: This is the code that needs parallelization.
        # TODO STEP 2: Try to read the values from the redis_client cache. If they do not exist, calculate and write.
        for program_id, obs in parsed_observations:
            # Check for a base target in the observation: if there is none, we cannot process.
            # For ToOs, this may be the case.
            base: Optional[Target] = obs.base_target()
            if base is None:
                logger.error(f'Could not find base target for {obs.id.id}.')
                continue

            program = Collector.get_program(program_id)
            if program is None:
                raise RuntimeError(f'Could not find program {program_id.id} for observation {obs.id.id}.')

            # Record the observation and target for this obs id.
            Collector._observations[obs.id] = obs, base

            # Compute the timing window expansion for the observation.
            # Then, calculate the target information, which performs the visibility calculations.
            tw = self._process_timing_windows(program, obs)
            ti = self._calculate_target_info(obs, base)
            Collector._target_info[base.name, obs.id] = ti

    def load_target_info_for_too(self, obs: Observation, target: Target) -> None:
        ti = self._calculate_target_info(obs, target)
        Collector._target_info[target.name, obs.id] = ti

    def night_configurations(self,
                             site: Site,
                             night_indices: NightIndices) -> Dict[NightIndices, NightConfiguration]:
        """
        Return the list of NightConfiguration for the site and nights under configuration.
        """
        return {night_idx: Collector._resource_service.get_night_configuration(
            site,
            self.get_night_events(site).time_grid[night_idx].datetime.date() - Day
        ) for night_idx in night_indices}

    def _get_group(self, obs: Observation) -> Group:
        """Return the group that an observation is a member of."""
        # TODO: How do we handle nested scheduling groups? Right now, if in a subgroup of a scheduling group, will fail.
        program = self.get_program(obs.belongs_to)
        # print(program.id)

        # Look for obs in the specified group. Compare by ID to avoid comparing full objects.
        def find_obs(g: Group) -> bool:
            return any(obs.unique_id == group_obs.unique_id for group_obs in g.observations())

        for group in program.root_group.children:
            if group.is_scheduling_group():
                for subgroup in group.children:
                    if find_obs(subgroup):
                        return group
            else:
                if find_obs(group):
                    return group

        # This should never happen: cannot find observation in program.
        raise RuntimeError(f'Could not find observation {obs.id.id} in program {program.id.id}.')

    def time_accounting(self,
                        plans: Plans,
                        sites: FrozenSet[Site] = ALL_SITES,
                        end_timeslot_bounds: Optional[Dict[Site, Optional[TimeslotIndex]]] = None) -> None:
        """
        For the given plans, which contain a set of plans for all sites for one night,
        perform time accounting on the plans for the specified sites up until the specified
        end timeslot for the site.

        If the end timeslot bound occurs during a visit, charge up to that timeslot
        For now, scheduling groups are charged only if they can be done completely.

        If end_timeslot_idx is not specified or not specified for a given site,
        then we perform time accounting across the entire night.
        """
        # Avoids repeated conversions in loop.
        time_slot_length = self.time_slot_length.to_datetime()

        for plan in plans:
            if plan.site not in sites:
                continue

            # Determine the end timeslot for the site if one is specified.
            # We set to None is the whole night is to be done.
            end_timeslot_bound = end_timeslot_bounds.get(plan.site) if end_timeslot_bounds is not None else None

            grpvisits = []
            # Restore this if we actually need ii, but seems it was just being used to check that grpvisits nonempty.
            # for ii, visit in enumerate(sorted(plan.visits, key=lambda v: v.start_time_slot)):
            for visit in sorted(plan.visits, key=lambda v: v.start_time_slot):
                obs = self.get_observation(visit.obs_id)
                group = self._get_group(obs)
                if grpvisits and group.is_scheduling_group() and group == grpvisits[-1].group:
                    grpvisits[-1].visits.append(visit)
                else:
                    grpvisits.append(GroupVisits(group=group, visits=[visit]))

            for grpvisit in grpvisits:
                # print(grpvisit.group.unique_id.id, grpvisit.start_time_slot(), grpvisit.end_time_slot())
                # Determine if group should be charged
                if grpvisit.group.is_scheduling_group():
                    # For now, only change aa scheduling group if it can be done fully
                    charge_group = end_timeslot_bound is None or end_timeslot_bound > grpvisit.end_time_slot()
                else:
                    charge_group = end_timeslot_bound is None or end_timeslot_bound > grpvisit.start_time_slot()

                # Charge if the end slot is less than this
                if end_timeslot_bound is not None:
                    end_timeslot_charge = end_timeslot_bound
                else:
                    end_timeslot_charge = grpvisit.end_time_slot() + 1

                # Charge to not_charged if the bound occurs during an AND (scheduling) group
                # TODO: for NIR + telluric, check if the standard was taken before the event, if so then charge for
                # what was observed and make a new copy of the telluric
                not_charged = (grpvisit.group.is_scheduling_group() and
                               grpvisit.start_time_slot() <= end_timeslot_charge <= grpvisit.end_time_slot())
                # print(f'charge_group = {charge_group}, charge_unused = {not_charged}')

                # print(f'\tGroup observations')
                # prog_obs = grpvisit.group.program_observations()
                part_obs = grpvisit.group.partner_observations()
                # print(f'\t\t Science')
                # for obs in prog_obs:
                #     print(f'\t\t {obs.unique_id.id}')
                # print(f'\t\t Partner')
                # for obs in part_obs:
                #     print(f'\t\t {obs.unique_id.id}')

                # print(f'\tVisits scheduled')
                for visit in grpvisit.visits:
                    # print(
                    #     f'\t\t{visit.obs_id.id} {visit.atom_start_idx} {visit.atom_end_idx} {visit.start_time_slot} '
                    #     f'{visit.time_slots} {visit.start_time_slot + visit.time_slots - 1}')

                    # Observation information
                    observation = self.get_observation(visit.obs_id)

                    # Number of slots in acquisition
                    n_slots_acq = time2slots(time_slot_length, observation.acq_overhead)
                    # print(f'\t\t{observation.acq_overhead.total_seconds()} {n_slots_acq}')

                    # Cumulative exec_times of unobserved atoms
                    cumul_seq = observation.cumulative_exec_times()
                    obs_seq = observation.sequence

                    # Check if the Observation has been completely observed.
                    if charge_group and visit.atom_end_idx == len(obs_seq) - 1:
                        logger.debug(f'Marking observation complete: {observation.id.id}')
                        observation.status = ObservationStatus.OBSERVED
                        if observation in part_obs:
                            part_obs.remove(observation)
                    elif not_charged:
                        observation.status = ObservationStatus.ONGOING

                    # Loop over atoms
                    for atom_idx in range(visit.atom_start_idx, visit.atom_end_idx + 1):
                        # calculate end time slot for each atom and compare with end_timeslot_charge
                        slot_length_visit = n_slots_acq + time2slots(time_slot_length, cumul_seq[atom_idx])  # noqa
                        slot_atom_end = visit.start_time_slot + slot_length_visit - 1

                        if atom_idx == visit.atom_start_idx:
                            slot_atom_length = slot_length_visit
                        else:
                            time_slots = time2slots(time_slot_length, cumul_seq[atom_idx-1])  # noqa
                            slot_atom_length = slot_length_visit - n_slots_acq - time_slots
                        if slot_atom_length > 0:
                            slot_atom_start = slot_atom_end - slot_atom_length + 1
                        else:
                            slot_atom_start = slot_atom_end - slot_atom_length

                        if slot_atom_end < end_timeslot_charge:
                            if charge_group:
                                # Charge to program or partner
                                # print(f'\t\t Charging program/partner times')
                                obs_seq[atom_idx].program_used = obs_seq[atom_idx].prog_time
                                obs_seq[atom_idx].partner_used = obs_seq[atom_idx].part_time

                                # Charge acquisition to the first atom.
                                if atom_idx == visit.atom_start_idx:
                                    if observation.obs_class == ObservationClass.PARTNERCAL:
                                        obs_seq[atom_idx].program_used += observation.acq_overhead
                                    elif (observation.obs_class == ObservationClass.SCIENCE or
                                          observation.obs_class == ObservationClass.PROGCAL):
                                        obs_seq[atom_idx].program_used += observation.acq_overhead

                                obs_seq[atom_idx].observed = True
                                obs_seq[atom_idx].qa_state = QAState.PASS

                            elif not_charged:
                                # charge to not_charged
                                not_charged_time = (end_timeslot_charge -
                                                    slot_atom_start + 1) * self.time_slot_length.to_datetime()
                                obs_seq[atom_idx].not_charged += not_charged_time

                # If charging the groups, set remaining partner cals to INACTIVE
                if charge_group:
                    for obs in part_obs:
                        # print(f'\t Setting {obs.unique_id.id} to INACTIVE.')
                        logger.debug(f'\tTime_accounting setting {obs.unique_id.id} to INACTIVE.')
                        obs.status = ObservationStatus.INACTIVE
