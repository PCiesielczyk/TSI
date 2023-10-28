class ClassMapper:
    def __init__(self):
        self.classes = {
            1: 'Speed limit (20km/h)',
            2: 'Speed limit (30km/h)',
            3: 'Speed limit (50km/h)',
            4: 'Speed limit (60km/h)',
            5: 'Speed limit (70km/h)',
            6: 'Speed limit (80km/h)',
            7: 'End of speed limit (80km/h)',
            8: 'Speed limit (100km/h)',
            9: 'Speed limit (120km/h)',
            10: 'No passing',
            11: 'No passing veh over 3.5 tons',
            12: 'Priority road',
            13: 'Stop',
            14: 'No vehicles',
            15: 'Veh > 3.5 tons prohibited',
            16: 'No entry',
            17: 'Bumpy road',
            18: 'Road work',
            19: 'End speed + passing limits',
            20: 'Turn right ahead',
            21: 'Turn left ahead',
            22: 'Ahead only',
            23: 'Go straight or right',
            24: 'Go straight or left',
            25: 'Keep right',
            26: 'Keep left',
            27: 'Roundabout mandatory',
            28: 'End of no passing',
            29: 'End no passing vehicle with a weight greater than 3.5 tons'
        }

    def map_to_text(self, numeric_class):
        return self.classes.get(numeric_class, "Unknown class")
