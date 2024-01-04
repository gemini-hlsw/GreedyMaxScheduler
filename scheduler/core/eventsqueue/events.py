# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from abc import ABC
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import final, FrozenSet, Optional

from lucupy.minimodel import Resource, Conditions, Site, TimeslotIndex
from lucupy.timeutils import time2slots


@dataclass
class Event(ABC):
    """
    Superclass for all events, i.e. Interruption and Blockage.
    """
    start: datetime
    reason: str
    site: Site

    def to_timeslot_idx(self, twi_eve_time: datetime, time_slot_length: timedelta) -> TimeslotIndex:
        """
        Given an event, calculate the timeslot offset it falls into relative to another datetime.
        This would typically be the twilight of the night on which the event occurs, hence the name twi_eve_time.
        """
        time_from_twilight = self.start - twi_eve_time
        time_slots_from_twilight = time2slots(time_slot_length, time_from_twilight)
        return TimeslotIndex(time_slots_from_twilight)


@dataclass
class Interruption(Event, ABC):
    """
    Parent class for any interruption that might cause a new schedule to be created.
    """
    ...


@dataclass
class Twilight(Interruption, ABC):
    """
    An event indicating that the 12 degree starting twilight for a night has been reached.
    """
    ...


@final
@dataclass
class EveningTwilight(Twilight):
    """
    An event indicating that the 12 degree starting twilight for a night has been reached.
    """
    ...


@final
@dataclass
class MorningTwilight(Twilight):
    """
    An event indicating that the 12 degree morning twilight for a night has been reached.
    This is used to finalize the time accounting for the night.
    """
    ...


@final
@dataclass
class WeatherChange(Interruption):
    """
    Interruption that occurs when new weather conditions come in.
    """
    new_conditions: Conditions


@dataclass
class Blockage(Event, ABC):
    """
    Parent class for any interruption that causes a blockage and requires a resume event.
    """
    end: Optional[datetime] = None  # needs a resume event

    def ends(self, end: datetime) -> None:
        self.end = end

    def time_loss(self) -> timedelta:
        if self.end:
            return self.end - self.start
        else:
            raise ValueError("Can't calculate Blockage time loss without end value.")


@dataclass
class ResumeNight(Interruption):
    """
    Event that lets the scheduler knows that the night can be resumed.
    """
    ...


@final
class Fault(Blockage):
    """
    Blockage that occurs when one or more resources experience a fault.
    """
    affects: FrozenSet[Resource]
