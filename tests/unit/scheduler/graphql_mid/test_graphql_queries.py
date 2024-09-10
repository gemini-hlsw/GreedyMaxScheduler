# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
import io

import pytest

from lucupy.observatory.abstract import ObservatoryProperties
from lucupy.observatory.gemini import GeminiProperties

from scheduler.graphql_mid.server import schema


@pytest.mark.asyncio
async def test_schedule_query_required_only():
    ObservatoryProperties.set_properties(GeminiProperties)
    query = """
        query Schedule {
            testSubQuery(scheduleId: "1", 
                         newScheduleInput: {startTime: "2018-10-01 08:00:00",
                                            endTime: "2018-10-03 08:00:00"
                                            sites: "GN", 
                                            mode: VALIDATION,
                                            semesterVisibility:false, 
                                            numNightsToSchedule:1})
        }
    """
    result = await schema.execute(query)
    assert result.errors is None


@pytest.mark.asyncio
async def test_schedule_query_with_all():
    ObservatoryProperties.set_properties(GeminiProperties)
    query = """
        query Schedule($programFile: Upload) {
            testSubQuery(scheduleId: "1", 
                         newScheduleInput: {startTime: "2018-10-01 08:00:00",
                                            endTime: "2018-10-03 08:00:00"
                                            sites: "GN", 
                                            mode: VALIDATION,
                                            semesterVisibility:false, 
                                            numNightsToSchedule:1,
                                            thesisFactor: 2.1,
                                            power: 3,
                                            metPower: 2.334,
                                            visPower: 3.222,
                                            whaPower: 2.0,
                                            programFile:$programFile})
        }
    """
    # Create a mock file
    mock_file = io.BytesIO(b"GN-2018B-Q-101")
    mock_file.name = "programs_ids.test.txt"
    variables = {"file": mock_file}

    result = await schema.execute(query, variable_values=variables)
    assert result.errors is None


@pytest.mark.asyncio
async def test_schedule_query_with_empty_file():
    ObservatoryProperties.set_properties(GeminiProperties)
    query = """
        query Schedule($programFile: Upload) {
            testSubQuery(scheduleId: "1", 
                         newScheduleInput: {startTime: "2018-10-01 08:00:00",
                                            endTime: "2018-10-03 08:00:00"
                                            sites: "GN", 
                                            mode: VALIDATION,
                                            semesterVisibility:false, 
                                            numNightsToSchedule:1,
                                            programFile:$programFile})
        }
    """
    # Create a mock file
    mock_file = io.BytesIO(b"")
    mock_file.name = "programs_ids.test.txt"
    variables = {"file": mock_file}

    result = await schema.execute(query, variable_values=variables)
    assert result.errors is None


@pytest.mark.asyncio
async def test_schedule_query_with_wrong_parameters():
    ObservatoryProperties.set_properties(GeminiProperties)
    query = """
        query Schedule {
            testSubQuery(scheduleId: "1", 
                         newScheduleInput: {startTime: "2018-10-01 08:00:00",
                                            endTime: "2018-10-03 08:00:00"
                                            sites: "GN", 
                                            mode: VALIDATION,
                                            semesterVisibility:false, 
                                            numNightsToSchedule:1,
                                            thesisFactor: 2.1,
                                            power: 3,
                                            metPower: 2.334,
                                            visPower: 3.222,
                                            whaPower: 2.0})
        }
    """

    result = await schema.execute(query)
    assert result.data is None
