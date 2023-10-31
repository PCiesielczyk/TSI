class ClassMapper:
    def __init__(self):
        self.classes = {
            1: 'Speed limit (20km/h)',
            2: 'Speed limit (30km/h)',
            3: 'Speed limit (40km/h)',
            4: 'Speed limit (50km/h)',
            5: 'Speed limit (60km/h)',
            6: 'Speed limit (70km/h)',
            7: 'Speed limit (80km/h)',
            8: 'End of speed limit (80km/h)',
            9: 'Speed limit (100km/h)',
            10: 'Speed limit (120km/h)',
            11: 'No passing',
            12: 'No passing veh over 3.5 tons',
            13: 'Stop',
            14: 'No vehicles',
            15: 'Veh > 3.5 tons prohibited',
            16: 'Bicycles prohibited',
            17: 'Give way to oncoming traffic',
            18: 'No left turn',
            19: 'No right turn',
            20: 'No Stopping',
            21: 'No Parking',
            22: 'No entry',
            23: 'Priority road',
            24: 'Turn right ahead',
            25: 'Turn left ahead',
            26: 'Ahead only',
            27: 'Go straight or right',
            28: 'Go straight or left',
            29: 'Keep right',
            30: 'Keep left',
            31: 'Roundabout mandatory',
            32: 'Road for bicycles',
            33: 'Priority for oncoming traffic',
            34: 'One-way road',
            35: 'Residential Zone',
            36: 'Parking',
            37: 'Restaurant',
            38: 'Bicycle Crossing',
            39: 'Bus stop',
            40: 'Pedestrian Crossing',
            41: 'Road without passing',
            42: 'End of the road for bicycles',
            43: 'End of the residential zone',
            44: 'End speed + passing limits',
            45: 'End of no passing',
            46: 'End no passing vehicle with a weight greater than 3.5 tons'
        }

    def map_to_text(self, numeric_class):
        return self.classes.get(numeric_class, "Unknown class")
