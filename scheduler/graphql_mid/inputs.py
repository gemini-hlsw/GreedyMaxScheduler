# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from datetime import datetime

import strawberry  # noqa
from strawberry.file_uploads import Upload  # noqa
from typing import Optional, List

from .scalars import Sites
from scheduler.core.builder.modes import SchedulerModes


__all__ = [
    'CreateNewScheduleInput',
    'UseFilesSourceInput',
]


@strawberry.input
class CreateNewScheduleInput:
    """
    Input for creating a new schedule.
    """
    start_time: str
    end_time: str
    sites: Sites
    mode: SchedulerModes
    semester_visibility: bool = True
    num_nights_to_schedule: Optional[int] = None
    thesis_factor: Optional[float] = 1.1
    power: Optional[int] = 2
    met_power: Optional[float] = 1.0
    vis_power: Optional[float] = 1.0
    wha_power: Optional[float] = 1.0
    program_file: Optional[Upload] = None


@strawberry.input
class UseFilesSourceInput:
    service: str
    sites: Sites
    calendar: Optional[Upload] = None
    gmos_fpus: Optional[Upload] = None
    gmos_gratings: Optional[Upload] = None
    faults: Optional[Upload] = None
    eng_tasks: Optional[Upload] = None


@strawberry.input
class NewFault:
    reason: str
    instrument: str  # change to enum
    start: datetime  # for Fault event
    end: datetime  # for ResumeNight event


@strawberry.input
class AddEventInput:
    events: List[NewFault]
