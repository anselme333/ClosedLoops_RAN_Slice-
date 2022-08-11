import veroviz as vrv
import pandas as pd
import matplotlib.pyplot as plt
import gmplot
import webbrowser
# Driving and flying region in Montreal
# Bounding Region in Montreal

# The inital source codes are from https://veroviz.org/
#  Modified by Anselme Ndikumana
print(vrv.checkVersion())
# Define a region within which the depot will be generated.
# These coordinates were obtained using Sketch (https://veroviz.org.sketch).

# Bounding Region:
myBoundingRegion = [[45.51206223390805, -73.55712889555436], [45.51185184184968, -73.54940413359147],
                    [45.50391648147121, -73.55378149870377], [45.50641143124836, -73.56369494322281]]

# Nodes:
nodesArray = [
    {'id': 0, 'lat': 45.510865806356136, 'lon': -73.55517804594001, 'altMeters': 0.0, 'nodeName': '1',
     'nodeType': 'depot', 'popupText': '1', 'leafletIconPrefix': 'glyphicon', 'leafletIconType': 'info-sign',
     'leafletColor': 'blue', 'leafletIconText': '0', 'cesiumIconType': 'pin', 'cesiumColor': 'blue',
     'cesiumIconText': '0', 'elevMeters': None},
    {'id': 1, 'lat': 45.51081696437442, 'lon': -73.5554623600956, 'altMeters': 0.0, 'nodeName': '2', 'nodeType': 'customer',
     'popupText': '2', 'leafletIconPrefix': 'glyphicon', 'leafletIconType': 'info-sign', 'leafletColor': 'blue',
     'leafletIconText': '1', 'cesiumIconType': 'pin', 'cesiumColor': 'blue', 'cesiumIconText': '1', 'elevMeters': None},
    {'id': 2, 'lat': 45.51118891378291, 'lon': -73.55431705684624, 'altMeters': 0.0, 'nodeName': '3',
     'nodeType': 'customer', 'popupText': '3', 'leafletIconPrefix': 'glyphicon', 'leafletIconType': 'info-sign',
     'leafletColor': 'blue', 'leafletIconText': '2', 'cesiumIconType': 'pin', 'cesiumColor': 'blue',
     'cesiumIconText': '2', 'elevMeters': None},
    {'id': 3, 'lat': 45.5111966461014, 'lon': -73.55283915939255, 'altMeters': 0.0, 'nodeName': '4', 'nodeType': 'customer',
     'popupText': '4', 'leafletIconPrefix': 'glyphicon', 'leafletIconType': 'info-sign', 'leafletColor': 'blue',
     'leafletIconText': '3', 'cesiumIconType': 'pin', 'cesiumColor': 'blue', 'cesiumIconText': '3', 'elevMeters': None},
    {'id': 4, 'lat': 45.50570461740554, 'lon': -73.55425000162087, 'altMeters': 0.0, 'nodeName': '5',
     'nodeType': 'customer', 'popupText': '5', 'leafletIconPrefix': 'glyphicon', 'leafletIconType': 'info-sign',
     'leafletColor': 'blue', 'leafletIconText': '4', 'cesiumIconType': 'pin', 'cesiumColor': 'blue',
     'cesiumIconText': '4', 'elevMeters': None},
    {'id': 5, 'lat': 45.50481472841, 'lon': -73.5534462327678, 'altMeters': 0.0, 'nodeName': '6', 'nodeType': 'customer',
     'popupText': '6', 'leafletIconPrefix': 'glyphicon', 'leafletIconType': 'info-sign', 'leafletColor': 'blue',
     'leafletIconText': '5', 'cesiumIconType': 'pin', 'cesiumColor': 'blue', 'cesiumIconText': '5', 'elevMeters': None},
]
myNodes = pd.DataFrame(nodesArray)
print("O-RUs", myNodes)
"""
ORUs = gmplot.GoogleMapPlotter(45.510865806356136, -73.55517804594001, 18)
ORUs.heatmap(myNodes['lat'], myNodes['lon'])
ORUs.draw("map.html")
webbrowser.open_new_tab("map.html")
OSRM-online, Modern C++ routing engine for shortest paths in road networks.
"""
vrv.createLeaflet(nodes=myNodes, boundingRegion=myBoundingRegion)
[time, dist] = vrv.getTimeDist2D(nodes=myNodes, matrixType='all2all', routeType='fastest', dataProvider='OSRM-online')

# Display as a nicely-formatted matrix for ground-based vehicles:
distance = vrv.convertMatricesDictionaryToDataframe(dist)
time_to_navigate = vrv.convertMatricesDictionaryToDataframe(time)
print("destination in meters ", distance)
print("time in seconds", time_to_navigate)
distance_between_ORUs_GroundVehicle = pd.DataFrame(distance)
distance_between_ORUs_GroundVehicle.to_csv('dataset/distance_between_ORUs_GroundVehicle.csv')

time_between_ORUs_GroundVehicle = pd.DataFrame(time_to_navigate)
time_between_ORUs_GroundVehicle.to_csv('dataset/time_between_ORUs_GroundVehicle.csv')




# Display as a nicely-formatted matrix for flying vehicles:

[totalTime, totalGroundDistance, totalFlightDistance] = vrv.getTimeDist3D(nodes=myNodes, matrixType='all2all',
                                                                          routeType='square', cruiseAltMetersAGL=120,
                                                                          takeoffSpeedMPS=5, cruiseSpeedMPS=12,
                                                                          landSpeedMPS=2, outputDistUnits='meters',
                                                                          outputTimeUnits='seconds')
distance_flying = vrv.convertMatricesDictionaryToDataframe(totalFlightDistance)
time_to_navigate_flying = vrv.convertMatricesDictionaryToDataframe(totalTime)
totalGroundDistance = vrv.convertMatricesDictionaryToDataframe(totalGroundDistance)

distance_between_ORUs_FlyingVehicle = pd.DataFrame(distance_flying)
distance_between_ORUs_FlyingVehicle.to_csv('dataset/distance_between_ORUs_FlyingVehicle.csv')

time_between_ORUs_FlyingVehicle = pd.DataFrame(time_to_navigate_flying)
time_between_ORUs_FlyingVehicle.to_csv('dataset/time_between_ORUs_FlyingVehicle.csv')

flying_ground_distance = pd.DataFrame(totalGroundDistance)
flying_ground_distance.to_csv('dataset/flying_ground_distance .csv')


print("destination in meters flying ", distance_flying)
print("time in seconds flying", time_to_navigate_flying)
print("totalGroundDistance", totalGroundDistance)


def SimpleRouting(nodesDF, dist, time):
    # Assume truck as ground vehicle  travels  from 0 -> 1 -> 2 ->3 -> 4 ->5-> 0
    # Assume drone as flying vehicle travels from 0 -> 1 -> 2 ->3 -> 4 ->5-> 0
    route = {'truck': [0, 1, 2, 3, 4, 5, 0],
             'drone': [0, 1, 2, 3, 4, 5, 0]}

    configs = {'truck': {
        'vehicleModels': ['veroviz/models/ub_truck.gltf'],
        'leafletColor': 'blue',
        'cesiumColor': 'Cesium.Color.BLUE',
        'packageModel': 'veroviz/models/box_blue.gltf'},
        'drone': {'vehicleModels': ['veroviz/models/drone.gltf',
                                    'veroviz/models/drone_package.gltf'],
                  'leafletColor': 'orange',
                  'cesiumColor': 'Cesium.Color.ORANGE',
                  'packageModel': 'veroviz/models/box_yellow.gltf'}}

    # Specify a duration for each vehicle to stop at each node.
    serviceTime = 0  # seconds/ we consider each node as O-RU. We assume that each vehicle does not stop at O-RU 

    # Initialize an empty "assignments" dataframe.
    vehicle_movement_df = vrv.initDataframe('assignments')

    for vehicle in route:
        startTime = 0
        for i in list(range(0, len(route[vehicle]) - 1)):
            startNode = route[vehicle][i]
            endNode = route[vehicle][i + 1]

            startLat = nodesDF[nodesDF['id'] == startNode]['lat'].values[0]
            startLon = nodesDF[nodesDF['id'] == startNode]['lon'].values[0]
            endLat = nodesDF[nodesDF['id'] == endNode]['lat'].values[0]
            endLon = nodesDF[nodesDF['id'] == endNode]['lon'].values[0]

            if ((vehicle == 'drone') and (startNode == 0)):
                # Use the 3D model of a drone carrying a package
                myModel = configs[vehicle]['vehicleModels'][1]
            else:
                # Use the 3D model of either a delivery truck or an empty drone
                myModel = configs[vehicle]['vehicleModels'][0]

            if (vehicle == 'truck'):
                # Get turn-by-turn navigation
                # for the truck on the road:
                shapepointsDF = vrv.getShapepoints2D(
                    objectID=vehicle,
                    modelFile=myModel,
                    startTimeSec=startTime,
                    startLoc=[startLat, startLon],
                    endLoc=[endLat, endLon],
                    routeType='fastest',
                    leafletColor=configs[vehicle]['leafletColor'],
                    cesiumColor=configs[vehicle]['cesiumColor'],
                    dataProvider='OSRM-online')
            else:
                # Get a 3D flight profile for the drone:
                shapepointsDF = vrv.getShapepoints3D(
                    objectID=vehicle,
                    modelFile=myModel,
                    startTimeSec=startTime,
                    startLoc=[startLat, startLon],
                    endLoc=[endLat, endLon],
                    takeoffSpeedMPS=5,
                    cruiseSpeedMPS=20,
                    landSpeedMPS=3,
                    cruiseAltMetersAGL=100,
                    routeType='square',
                    cesiumColor=configs[vehicle]['cesiumColor'])

                # Add the vehicle movement to the assignments dataframe:
            vehicle_movement_df = pd.concat([vehicle_movement_df, shapepointsDF],
                                      ignore_index=True, sort=False)

            # Update the time
            startTime = max(shapepointsDF['endTimeSec'])

            # Add loitering for service
            vehicle_movement_df = vrv.addStaticAssignment(
                initAssignments=vehicle_movement_df,
                objectID=vehicle,
                modelFile=myModel,
                loc=[endLat, endLon],
                startTimeSec=startTime,
                endTimeSec=startTime + serviceTime)

            # Update the time again
            startTime = startTime + serviceTime

            # Leave a package at all non-depot nodes:
            if (endNode != 0):
                vehicle_movement_df = vrv.addStaticAssignment(
                    initAssignments=vehicle_movement_df,
                    objectID='package %d' % endNode,
                    modelFile=configs[vehicle]['packageModel'],
                    loc=[endLat, endLon],
                    startTimeSec=startTime,
                    endTimeSec=-1)

    return vehicle_movement_df


vehicle_movement_df = SimpleRouting(myNodes, dist, time)
vrv.createLeaflet(nodes=myNodes, arcs=vehicle_movement_df)
vehicle_movement_df = vehicle_movement_df[vehicle_movement_df.columns[~vehicle_movement_df.columns.isin(
    ['modelFile', 'modelScale', 'modelMinPxSize','leafletColor', 'leafletWeight', 'leafletStyle', 'leafletOpacity',
     'leafletCurveType', 'leafletCurvature', 'useArrows', 'cesiumColor', 'cesiumWeight', 'cesiumStyle', 'cesiumOpacity',
     'ganttColor', 'popupText', 'startElevMeters', 'endElevMeters',  'wayname', 'waycategory', 'surface', 'waytype',
     'steepness', 'tollway'])]]
print(vehicle_movement_df)
print(vehicle_movement_df.columns)
vehicle_movement_df = vehicle_movement_df[vehicle_movement_df.endTimeSec != -1]
vehicle_movement_df.to_csv('dataset/vehicle_movement.csv')
