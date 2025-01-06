class Values:
    def __init__(self):
        # ABSOLUTE MOUSE POSITION
        self.x_factor = 2000
        self.y_factor = 1200
        # DELTA JSON STATE
        self.delta_x_factor = 5000  # delta_position_x
        self.delta_y_factor = 5000  # delta_position_y
        self.delta_z_factor = 10 # delta_position_z
        self.delta_crossSectionScale_factor = 100  # delta_crossSectionScale
        self.delta_q1_factor = 1  # delta_projectionOrientation_q1
        self.delta_q2_factor = 1  # delta_projectionOrientation_q2
        self.delta_q3_factor = 1  # delta_projectionOrientation_q3
        self.delta_q4_factor = 1  # delta_projectionOrientation_q4
        self.delta_projectionScale_factor = 100  # delta_projectionScale

        # FOR THE POSITION STATE NORMALIZATION
        # From the FLywire dataset, we have the following values
        min_position= [20400, 5760, 16]
        max_position= [236800, 118400, 7062]
        max_crossSectionScale= 300
        max_projectionOrientation= [1, 1, 1, 1]
        max_projectionScale= 1000000 # this is a very high value

        self.position_x_factor =  max_position[0]-min_position[0]
        self.position_y_factor =  max_position[1]-min_position[1]
        self.position_z_factor =  max_position[2]-min_position[2]
        self.crossSectionScale_factor = max_crossSectionScale
        self.projectionOrientation_q1_factor = max_projectionOrientation[0]
        self.projectionOrientation_q2_factor = max_projectionOrientation[1]
        self.projectionOrientation_q3_factor = max_projectionOrientation[2]
        self.projectionOrientation_q4_factor = max_projectionOrientation[3]
        self.projectionScale_factor = max_projectionScale
    