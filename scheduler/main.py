# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
import asyncio
import json
import os

import uvicorn
from astropy.time import Time
from fastapi.responses import JSONResponse
from fastapi import WebSocket, WebSocketDisconnect
from lucupy.minimodel import ALL_SITES
from lucupy.observatory.gemini.geminiproperties import GeminiProperties  # isort: skip
from lucupy.observatory.abstract import ObservatoryProperties  # isort: skip
from fastapi.responses import HTMLResponse
from starlette.requests import Request

from scheduler.app import app
from scheduler.config import config
from scheduler.connection_manager import ConnectionManager
from scheduler.core.builder.modes import SchedulerModes
from scheduler.core.components.ranker import RankerParameters
from scheduler.core.service import Service
from scheduler.params import SchedulerParameters

heroku_port = os.environ.get("PORT")
manager = ConnectionManager()

# Root API
@app.get("/", include_in_schema=False)
def root() -> JSONResponse:
    return JSONResponse(status_code=200,
                        content={
                            "message": "Welcome to Server"})


@app.websocket("/ws/{client_id}")
async def schedule_websocket(websocket: WebSocket, client_id: int):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            if data:
                for n in Service().generate(**SchedulerParameters.from_json(data).__dict__):
                    await manager.send(n, websocket)
            else:
                raise ValueError('Missing parameters to create schedule')
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.get("/websocket-client")
async def websocket_client(request: Request):
    return HTMLResponse("""
<!DOCTYPE html>
<html>
    <head>
        <title>Schedule</title>
    </head>
    <body>
        <h1>WebSocket Plans</h1>
        <h2>Your ID: <span id="ws-id"></span></h2>
        <form action="" onsubmit="sendMessage(event)">
            <label for="startTime">Start:</label>
            <input type="text" id="startTime" value="2018-10-01 08:00:00" autocomplete="off"/><br/>
            <label for="endTime">End:</label>
            <input type="text" id="endTime" value="2018-10-03 08:00:00" autocomplete="off"/><br/>
            <label for="sites">Sites:</label>
            <input type="text" id="sites" value="GN" autocomplete="off"/><br/>
            <label for="mode">Mode:</label>
            <input type="text" id="mode" value="VALIDATION" autocomplete="off"/><br/>
            <label for="semesterVisibility">semester Visibility:</label>
            <input type="text" id="semesterVisibility" value="True" autocomplete="off"/><br/>
            <label for="numNightsToSchedule"> num nights to schedule:</label>
            <input type="text" id="numNightsToSchedule" value="3" autocomplete="off"/><br/>
            <label for="thesisFactor">thesis_actor:</label>
            <input type="number" id="thesisFactor" value="1.1" autocomplete="off"/><br/>
            <label for="power">power:</label>
            <input type="number" id="power" value="2.0" autocomplete="off"/><br/>
            <label for="metPower">met_power:</label>
            <input type="number" id="metPower" value="1.0" autocomplete="off"/><br/>
            <label for="visPower">vis_power:</label>
            <input type="number" id="visPower" value= "1.0" autocomplete="off"/><br/>
            <label for="whaPower">wha_power:</label>
            <input type="number" id="whaPower" value= "1.0" autocomplete="off"/><br/>
            <button>Run</button>
        </form>
        <ul id='messages'>
        </ul>
        <script>
            var client_id = Date.now()
            document.querySelector("#ws-id").textContent = client_id;
            var ws = new WebSocket(`ws://localhost:8000/ws/${client_id}`);
            
            ws.onmessage = function(event) {
                var messages = document.getElementById('messages')
                var message = document.createElement('li')
                var content = document.createTextNode(JSON.stringify(event.data, null, 2))
                message.appendChild(content)
                messages.appendChild(message)
            };
            function sendMessage(event) {
                
                var input1 = document.getElementById("startTime")
                var input2 = document.getElementById("endTime")
                var input3 = document.getElementById("sites")
                var input4 = document.getElementById("mode")
                var input5 = document.getElementById("semesterVisibility")
                var input6 = document.getElementById("numNightsToSchedule")
                var input7 = document.getElementById("thesisFactor")
                var input8 = document.getElementById("power")
                var input9 = document.getElementById("metPower")
                var input10 = document.getElementById("visPower")
                var input11 = document.getElementById("whaPower")
                
                var data = {
                    startTime: input1.value,
                    endTime: input2.value,
                    sites: [input3.value ],
                    schedulerMode: input4.value,
                    semesterVisibility: input5.value,
                    numNightsToSchedule: input6.value,
                    rankerParameters: {
                        thesisFactor: input7.value,
                        power: input8.value,
                        metPower: input9.value,
                        visPower: input10.value,
                        whaPower: input11.value,
                    }
                }
                ws.send(JSON.stringify(data));
                event.preventDefault()
            }
        </script>
    </body>
</html>
""")


def main():
    # Setup lucupy properties
    # TODO: This should be dynamic but since we are just working with Gemini right now
    #       should not be an issue.
    ObservatoryProperties.set_properties(GeminiProperties)
    uvicorn.run(app,
                host=config.server.host,
                port=heroku_port if heroku_port else config.server.port,
                ws_ping_interval=50)


if __name__ == "__main__":
    main()
