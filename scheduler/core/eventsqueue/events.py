# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import uuid
from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import final, FrozenSet

from lucupy.minimodel import Resource, Site, TimeslotIndex, VariantSnapshot, ObservationID, Target
from lucupy.timeutils import time2slots


__all__ = [
    'UUIDIdentified',
    'UUIDReferenced',
    'Event',
    'RoutineEvent',
    'TwilightEvent',
    'EveningTwilightEvent',
    'MorningTwilightEvent',
    'InterruptionEvent',
    'WeatherChangeEvent',
    'FaultEvent',
    'InterruptionResolutionEvent',
    'FaultResolutionEvent',
    'WeatherClosureEvent',
    'WeatherClosureResolutionEvent',
    'ToOActivationEvent'
]


# TODO: These can PROBABLY all be made frozen.

@dataclass(frozen=True)
class UUIDIdentified(ABC):
    """
    A class with an automatic UUID attached to it.
    TODO: This should be moved to lucupy.
    """
    id: uuid.UUID = field(default_factory=uuid.uuid4, init=False)

    def __eq__(self, other):
        if isinstance(other, UUIDIdentified):
            return self.id == other.id
        return False

    def __hash__(self):
        return hash(self.id)


@dataclass(frozen=True)
class UUIDReferenced(ABC):
    """
    A class for an object that maintains a reference to a UUIDIdentified object.
    """
    uuid_identified: UUIDIdentified

    @property
    def uuid_referenced(self) -> uuid:
        return self.uuid_identified.id


@dataclass(frozen=True)
class Event(UUIDIdentified, ABC):
    """
    Superclass for all events. They contain:
    1. The site at which the event occurred.
    2. The time (as a datetime object) at which an event occurred.
    3. A human-readable description of the event.
    """
    site: Site
    time: datetime
    description: str

    def to_timeslot_idx(self, twi_eve_time: datetime, time_slot_length: timedelta) -> TimeslotIndex:
        """
        Given an event, calculate the timeslot offset it falls into relative to another datetime.
        This would typically be the twilight of the night on which the event occurs, hence the name twi_eve_time.
        """
        time_from_twilight = self.time - twi_eve_time
        time_slots_from_twilight = time2slots(time_slot_length, time_from_twilight)
        return TimeslotIndex(time_slots_from_twilight)


@dataclass(frozen=True)
class RoutineEvent(Event, ABC):
    """
    A routine event that is predictable and processed by the Scheduler.
    Examples include evening and morning twilight.
    """
    ...


@dataclass(frozen=True)
class TwilightEvent(RoutineEvent, ABC):
    """
    An event indicating that the 12 degree starting twilight for a night has been reached.
    """
    ...


@final
@dataclass(frozen=True)
class EveningTwilightEvent(TwilightEvent):
    """
    An event indicating that the 12 degree starting twilight for a night has been reached.
    """
    ...


@final
@dataclass(frozen=True)
class MorningTwilightEvent(TwilightEvent):
    """
    An event indicating that the 12 degree morning twilight for a night has been reached.
    This is used to finalize the time accounting for the night.
    """
    ...


@dataclass(frozen=True)
class InterruptionEvent(Event, ABC):
    """
    Parent class for any interruption that might cause a new schedule to be created.
    These events include:
    1. Events that have no specified end time (e.g. weather changes).
    2. Events that have a specified end time (e.g. engineering tasks, faults) and thus are paired together
       with an InterruptionResolutionEvent.
    """
    ...


@final
@dataclass(frozen=True)
class WeatherChangeEvent(InterruptionEvent):
    """
    Interruption that occurs when new a new weather variant comes in.
    """
    variant_change: VariantSnapshot


@final
@dataclass(frozen=True)
class ToOActivationEvent(InterruptionEvent):
    """
    Change the status of a ToO from ON_HOLD to READY.
    """
    too_id: ObservationID
    target: Target

@final
@dataclass(frozen=True)
class FaultEvent(InterruptionEvent):
    """
    Interruption that occurs when there is a fault in a resource.
    In OCS, this will likely be the site itself where the fault occurred.
    """
    affects: FrozenSet[Resource]


@final
@dataclass(frozen=True)
class WeatherClosureEvent(InterruptionEvent):
    """
    A weather closure for a given site.
    This will be treated like a FaultEvent, but the "affects" Resource will be the entire site.
    """
    ...

    @property
    def affects(self) -> FrozenSet[Resource]:
        return frozenset([self.site.resource])


@dataclass(frozen=True)
class InterruptionResolutionEvent(Event, UUIDReferenced, ABC):
    """
    A class representing the resolution of an interruption that can be resolved (e.g. a resolved fault
    or the end of an engineering task.)

    These events, signifying the end of a period of time, can be used to generate a time loss.
    """
    @property
    def time_loss(self) -> timedelta:
        """
        Calculate the time loss from this InterruptionEvent to this InterruptionEventResolution as a timedelta.
        TODO: This assumes that Interruption events are notified or processed at Twilight (or near) situation that might
        TODO: be completely different when Resource is implemented and real input is processed.
        """
        if not isinstance(self.uuid_identified, InterruptionEvent):
            raise ValueError(f'{self.__class__.__name__} ({self.description}) does not have a corresponding '
                             'interruption event.')
        return self.time - self.uuid_identified.time

    def time_slot_loss(self, time_slot_length: timedelta) -> int:
        """
        Given the length of a time slot, calculate the number of time slots lost from the InterruptionEvent to
        this InterruptionEventResolution.
        """
        return time2slots(time_slot_length, self.time_loss)


@final
@dataclass(frozen=True)
class FaultResolutionEvent(InterruptionResolutionEvent, UUIDReferenced):
    """
    Interruption that occurs when a Fault is resolved.
    """
    def __post_init__(self):
        if not isinstance(self.uuid_identified, FaultEvent):
            raise ValueError('FaultResolutionEvent is not paired with a FaultEvent.')


@final
@dataclass(frozen=True)
class WeatherClosureResolutionEvent(InterruptionResolutionEvent, UUIDReferenced):
    """
    Interruption that occurs when a WeatherClosure is resolved.
    """
    def __post_init__(self):
        if not isinstance(self.uuid_identified, WeatherClosureEvent):
            raise ValueError('WeatherClosureResolutionEvent is not paired with a WeatherClosureEvent.')
