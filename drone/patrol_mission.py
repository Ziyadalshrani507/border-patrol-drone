import asyncio
import datetime
from mavsdk import System
from mavsdk.action import OrbitYawBehavior
from mavsdk.mission import MissionItem, MissionPlan


async def uav():

    # connect 
    drone = System()
    await drone.connect(system_address="udpin://0.0.0.0:14540")
    print("Waiting for connection...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Drone connected")
            break

    #  health check 
    print("Waiting for drone to be ready...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("Drone is ready!")
            break

    #  home position 
    home = await anext(drone.telemetry.home())
    lat  = home.latitude_deg
    lon  = home.longitude_deg
    alt  = home.absolute_altitude_m
    print(f"Home: lat={lat:.6f}, lon={lon:.6f}, alt={alt:.1f}m")

    # battery 
    battery = await anext(drone.telemetry.battery())
    print(f"Battery: {battery.remaining_percent * 10:.0f}%")

    #mission items 
    mission_items = [

        # W1
        MissionItem(18.708306, 49.963900,
                    30, 10, True,
                    float('nan'), float('nan'),
                    MissionItem.CameraAction.NONE,
                    float('nan'), float('nan'),
                    10, float('nan'),
                    float('nan'),
                    MissionItem.VehicleAction.NONE),

        # W2
        MissionItem(18.708280, 49.964400,
                    30, 10, True,
                    float('nan'), float('nan'),
                    MissionItem.CameraAction.NONE,
                    float('nan'), float('nan'),
                    10, float('nan'),
                    float('nan'),
                    MissionItem.VehicleAction.NONE),

        # W3
        MissionItem(18.708270, 49.964900,
                    30, 10, True,
                    float('nan'), float('nan'),
                    MissionItem.CameraAction.NONE,
                    float('nan'), float('nan'),
                    10, float('nan'),
                    float('nan'),
                    MissionItem.VehicleAction.NONE),

        # W4 — stop here orbit and report
        MissionItem(18.708260, 49.965400,
                    30, 10, False,
                    float('nan'), float('nan'),
                    MissionItem.CameraAction.NONE,
                    float('nan'), float('nan'),
                    10, float('nan'),
                    float('nan'),
                    MissionItem.VehicleAction.NONE),

        # W5
        MissionItem(18.708250, 49.965900,
                    30, 10, True,
                    float('nan'), float('nan'),
                    MissionItem.CameraAction.NONE,
                    float('nan'), float('nan'),
                    10, float('nan'),
                    float('nan'),
                    MissionItem.VehicleAction.NONE),

        # W6
        MissionItem(18.708250, 49.966400,
                    30, 10, True,
                    float('nan'), float('nan'),
                    MissionItem.CameraAction.NONE,
                    float('nan'), float('nan'),
                    10, float('nan'),
                    float('nan'),
                    MissionItem.VehicleAction.NONE),

        # W7
        MissionItem(18.708250, 49.966667,
                    30, 10, True,
                    float('nan'), float('nan'),
                    MissionItem.CameraAction.NONE,
                    float('nan'), float('nan'),
                    10, float('nan'),
                    float('nan'),
                    MissionItem.VehicleAction.NONE),
    ]

    #  upload mission 
    await drone.mission.set_return_to_launch_after_mission(True)
    mission_plan = MissionPlan(mission_items)
    await drone.mission.upload_mission(mission_plan)
    print("Mission uploaded")

    #  arming drone 
    await drone.action.arm()
    print("Drone armed")

    #  start mission 
    await drone.mission.start_mission()
    print("Mission started")
    print("="*50)

    #  monitor progress 
    async for progress in drone.mission.mission_progress():
        current = progress.current
        total   = progress.total

        print(f"\nMission progress: {current}/{total}")

        #  W4 — orbit and report 
        if current == 3:

            # pause mission
            await drone.mission.pause_mission()
            print("Suspicious activity detected at W4!")
            print("Pausing mission...")

            # get current position
            position = await anext(drone.telemetry.position())

            # do orbit — 
            print("Starting orbit...")
            await drone.action.do_orbit(
                radius_m            = 50,
                velocity_ms         = 10,
                yaw_behavior        = OrbitYawBehavior.HOLD_FRONT_TO_CIRCLE_CENTER,
                latitude_deg        = position.latitude_deg,
                longitude_deg       = position.longitude_deg,
                absolute_altitude_m = position.absolute_altitude_m
            )

            # orbit for 20 seconds
            print("Orbiting for 20 seconds...")
            await asyncio.sleep(20)

            # simulate AI detection
            print("Running AI detection...")
            await asyncio.sleep(3)

            # print report to terminal
            time_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print("")
            print("="*50)
            print("BORDER GUARD ALERT")
            print("="*50)
            print(f"Time       : {time_now}")
            print(f"Location   : W4 — border crossing hotspot")
            print(f"Lat        : {position.latitude_deg:.6f}")
            print(f"Lon        : {position.longitude_deg:.6f}")
            print(f"People     : 5 detected crossing the border")
            print(f"Vehicles   : 1 car detected crossing the border")
            print(f"Status     : IMMEDIATE ACTION REQUIRED")
            print("="*50)
            print("")

            # resume mission
            print("Resuming patrol...")
            await drone.mission.start_mission()
            await asyncio.sleep(2)

        
        if current == total:
            print("")
            print("="*50)
            print("PATROL COMPLETE")
            print("="*50)
            print("Drone returning to base")
            print("="*50)
            break


asyncio.run(uav())