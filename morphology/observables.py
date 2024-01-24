class Observables:
    def __init__(self):
        self.sensors = []

    def add_sensor(self, sensor):
        self.sensors.append(sensor)

    def remove_sensor(self, sensor):
        self.sensors.remove(sensor)

    def get_sensor_data(self):
        sensor_data = []
        for sensor in self.sensors:
            sensor_data.append(sensor.get_data())
        return sensor_data

class Sensor:
    def __init__(self, name):
        self.name = name

    def get_data(self):
        # Example implementation of getting sensor data
        return f"{self.name} data"
