# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import asyncio
import signal
import functools
from datetime import datetime
from multiprocessing import Process
from random import randint

from scheduler.core.meta import Singleton
from scheduler.core.service.service import Service
from scheduler.core.service.modes import SchedulerModes
from scheduler.config import config

from .runner import StandardRunner
from .task import SchedulerTask, TaskType

DEFAULT_TIMEOUT = 10  # seconds
DEFAULT_SIZE = 5  # number of tasks to run in parallel


class ProcessManager(metaclass=Singleton):
    """
    Main handler for each runner, which is responsible for scheduling the task.
    """

    def __init__(self, size: int = DEFAULT_SIZE, timeout: int = DEFAULT_TIMEOUT):
        self.realtime_runner = StandardRunner(1)
        self.standard_runner = StandardRunner(size)
        self.timeout = timeout

    def schedule_with_runner(self, task: SchedulerTask, mode: TaskType):
        """
        Schedule a task with the corresponding runner for the given mode.
        """
        if mode == TaskType.REALTIME:
            return self.realtime_runner.schedule(Process(target=task.target), task.timeout)
        elif mode == TaskType.STANDARD:
            return self.standard_runner.schedule(Process(target=task.target), task.timeout)
        else:
            raise ValueError(f'Invalid mode {mode}')

    def add_task(self, start: datetime, target: callable, mode: TaskType) -> None:
        task = SchedulerTask(start,
                             target,
                             self.timeout)
        self.schedule_with_runner(task, mode)

    async def run(self, scheduler: Service, period: int, mode: TaskType):
        done = asyncio.Event()

        def shutdown():
            done.set()
            self.shutdown()
            asyncio.get_event_loop().stop()

        asyncio.get_event_loop().add_signal_handler(signal.SIGINT, shutdown)

        while not done.is_set():
            self.add_task(datetime.now(), scheduler, mode)
            if period == 0:
                # random case #
                await asyncio.sleep(randint(1, 10))
            else:
                await asyncio.sleep(period)

    def shutdown(self):
        """
        Callback for shutting down the process manager.
        """
        self.realtime_runner.terminate_all()
        self.standard_runner.terminate_all()


def setup_with(mode: SchedulerModes):
    # Setup scheduler mode
    try:
        mode = SchedulerModes[config.mode.upper()]
    except KeyError:
        raise ValueError('Mode is Invalid!')

    def decorator_setup(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            pm = func(*args, **kwargs)
            if mode is SchedulerModes.OPERATION:
                pm.size = 1
            else:
                pm.size = config.process_manager.size
            return pm
        return wrapper
    return decorator_setup


@setup_with(config.mode)
def setup_manager():
    """Setup the manager based on the mode using setup_with decorator.

    Default values:
        TIMEOUT = 10 seconds
        SIZE = 5 task at the same time (Not valid for Operation).

    Returns:
        ProcessManager: Default Process Manager if timeout is not set.
    """
    if config.process_manager.timeout:
        return ProcessManager(timeout=config.process_manager.timeout)
    return ProcessManager()
