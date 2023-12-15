# Copyright (c) 2016-2023 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from abc import ABC
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import ceil
from typing import final, FrozenSet, Optional

from lucupy.minimodel import Resource, Conditions, Site, TimeslotIndex


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
        time_slots_from_twilight = ceil(time_from_twilight / time_slot_length)
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

@final
class Fault(Interruption):
    """
    Blockage that occurs when one or more resources experience a fault.
    """
    id: str
    end: datetime
    time_loss: timedelta
    affects: FrozenSet[Resource]


@final
class EngTask(Interruption):

    end: datetime
    time_loss: timedelta
