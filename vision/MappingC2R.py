import cv2
import numpy as np

class MappingC2R():
    """
    (input) image(pixel) point <-> checkerboard coordinate <-> robot's coordinate (output)
    """
    def __init__(self, filename_mapping_table, row, col):
        # checkerboard dimension
        self.__row = row
        self.__col = col

        # checkerboard corners
        filename = '/home/hwangmh/pycharmprojects/FLSpegtransfer/calibration_files/corners_8x6.npy'
        self.__cb_corners = np.load(filename).reshape(self.__row, self.__col, 2)

        # mapping table (cb corners <-> robot)
        self.mapping_table_0 = self.load_mapping_table(filename_mapping_table)
        filename_mapping_table_90 = filename_mapping_table + '_90'
        self.mapping_table_90 = self.load_mapping_table(filename_mapping_table_90)

    def load_mapping_table(self, filename):
        data_default = np.loadtxt(filename, delimiter=',')
        mapping_table = np.zeros((self.__row, self.__col, data_default.shape[1]-2))
        for line in data_default:
            arg_c = line[0].astype(int)
            arg_r = line[1].astype(int)
            mapping_table[arg_r][arg_c] = line[2:]
        return mapping_table

    def check_corners(self, input_pixel, polygon):
        check = False
        for p in polygon:
            if np.allclose(p,input_pixel):
                check = True
        return check

    def check_edges(self, input_pixel, polygon):
        check = False
        for i in range(len(polygon)):
            j = (i+1) % len(polygon);
            if (input_pixel[0] < polygon[i][0]) != (input_pixel[0] < polygon[j][0]):
                if (input_pixel[1] < polygon[i][1]) != (input_pixel[1] < polygon[j][1]):
                    a = polygon[j][0]-polygon[i][0]
                    b = polygon[i][1]-polygon[j][1]
                    c = a*polygon[i][1] + b*polygon[i][0]
                    if np.isclose(a*input_pixel[1] + b*input_pixel[0], c):
                        check = True
        return check

    def check_inside(self, input_pixel, polygon):
        cross = 0;  # number of points that the polygon crosses with right half-line
        for i in range(len(polygon)):
            j = (i+1) % len(polygon);
            if (input_pixel[0] < polygon[i][0]) != (input_pixel[0] < polygon[j][0]):
                # crossing point
                p = (polygon[j][1]- polygon[i][1]) * (input_pixel[0]-polygon[i][0]) / (polygon[j][0]-polygon[i][0])+polygon[i][1]
                if input_pixel[1] < p:
                    cross += 1
        return cross % 2 > 0;

    def find_around_corners(self, cb_corners, input_pixel):
        """
        Find four corners around the input pixel
        :param cb_corners: checkerboard coordinates
        :param input_pixel: input pixel surrounded by four corners
        :return: coordinate of the four corners and their arguments
        """

        for r in range(self.__row - 1):
            for c in range(self.__col-1):
                four_corners = [cb_corners[r][c], cb_corners[r+1][c], cb_corners[r+1][c+1], cb_corners[r][c+1]]
                arg = [[r,c],[r+1,c], [r+1,c+1],[r,c+1]]
                if self.check_corners(input_pixel, four_corners):
                    return four_corners, arg
                elif self.check_edges(input_pixel, four_corners):
                    return four_corners, arg
                elif self.check_inside(input_pixel, four_corners):
                    return four_corners, arg
        return []

    def bilinear_interpolation(self, x,y, x1,x2, y1,y2, fQ11, fQ12, fQ21, fQ22):
        """
        Perform bilinear interpolation between four corner points
        (x,y): input point
        (x1,x2), (y1,y2), (fQ11, fQ12, fQ21, fQ22): given points with its function value
        :return: interpolated value
        """
        assert x1!=x2
        assert y1!=y2
        if np.isclose(x, min(x1,x2)):   x=min(x1,x2)
        if np.isclose(x, max(x1,x2)):   x=max(x1,x2)
        if np.isclose(y, min(y1,y2)):   y=min(y1,y2)
        if np.isclose(y, max(y1,y2)):   y=max(y1,y2)
        assert min(x1,x2) <= x and x <= max(x1,x2)
        assert min(y1,y2) <= y and y <= max(y1,y2)
        output = 1/(x2-x1)/(y2-y1)*(fQ11*(x2-x)*(y2-y)+fQ21*(x-x1)*(y2-y)+fQ12*(x2-x)*(y-y1)+fQ22*(x-x1)*(y-y1))
        return output

    def rectify_points(self, pts_src, pts_input):
        pts_src = np.array(pts_src)
        pts_dst = np.array([[0, 0], [0, 100], [100, 100], [100, 0]], np.float64)

        # Calculate Homography
        h, _ = cv2.findHomography(pts_src, pts_dst)

        # choose center point as an input
        pts_input = np.append(pts_input, 1).reshape(3, 1)
        pts_rectified = np.matrix(h) * pts_input
        pts_rectified = np.array((pts_rectified/pts_rectified[2])[:2]).squeeze()
        return pts_dst, pts_rectified

    def transform_pixel2robot(self, input_pixel, roll_angle):
        surrounding_corners, arg = self.find_around_corners(self.__cb_corners, input_pixel)
        pts_dst, pts_input_rectified = self.rectify_points(surrounding_corners, input_pixel)
        x = pts_input_rectified[0]
        y = pts_input_rectified[1]
        x1 = pts_dst[0][0]  # =0
        x2 = pts_dst[3][0]  # =100
        y1 = pts_dst[0][1]  # =0
        y2 = pts_dst[1][1]  # =100
        fQ11 = self.mapping_table_0[arg[0][0], arg[0][1]]
        fQ12 = self.mapping_table_0[arg[1][0], arg[1][1]]
        fQ21 = self.mapping_table_0[arg[3][0], arg[3][1]]
        fQ22 = self.mapping_table_0[arg[2][0], arg[2][1]]
        output_0 = self.bilinear_interpolation((x,y), (x1,x2), (y1,y2), (fQ11,fQ12,fQ21,fQ22))

        fQ11 = self.mapping_table_90[arg[0][0], arg[0][1]]
        fQ12 = self.mapping_table_90[arg[1][0], arg[1][1]]
        fQ21 = self.mapping_table_90[arg[3][0], arg[3][1]]
        fQ22 = self.mapping_table_90[arg[2][0], arg[2][1]]
        output_90 = self.bilinear_interpolation((x,y), (x1,x2), (y1,y2), (fQ11,fQ12,fQ21,fQ22))

        # interpolate the outputs according to the given roll angle
        roll_angle = abs(roll_angle)
        if roll_angle > 90:
            roll_angle -= 90
            roll_angle = abs(roll_angle)
        output_final = output_0 + (roll_angle - 0) * (output_90 - output_0) / (90 - 0)
        return output_final

if __name__ == '__main__':
    mapping = MappingC2R(filename_mapping_table='../calibration_files/mapping_table_PSM1', row=6, col=8)
    pixel = [104,   94]
    pixel = [  76.96347809,  68.91957855]
    pixel = [442,334]
    theta = 45
    output = mapping.transform_pixel2robot(input_pixel=pixel, roll_angle=theta)

    # imgfile = "../img/img_color_checkerboard.png"
    # img_cb = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)
    # cv2.imshow("cb", img_cb)
    # cv2.waitKey(0)