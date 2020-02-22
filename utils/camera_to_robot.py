import numpy as np

camera_cloud = np.array([
	[0.0288291685283,-0.0516032800078,0.865170359612],
	[-0.00393536826596,-0.0540089458227,0.862051427364],
	[-0.0531960763037,-0.0516039654613,0.86183142662],
	[-0.0529234446585,-0.0899467468262,0.863367021084],
	[-0.0546798966825,-0.136061891913,0.860864520073],
	[-0.020501403138,-0.135024949908,0.864968299866],
	[-0.0202115140855,-0.103962011635,0.866482436657],
	[0.0285633616149,-0.0957856029272,0.867283821106],
	[0.0272694099694,-0.13405893743,0.866086661816],
	[0.0264145564288,-0.124313704669,0.835303366184],
	[0.0255187060684,-0.0763552412391,0.829324007034],
	[-0.00903954543173,-0.0993611812592,0.83287358284],
	[-0.0449533797801,-0.136231228709,0.848905026913],
	[-0.0524198338389,-0.0574322640896,0.837676882744],
	[0.0268572829664,-0.0617311745882,0.778766036034],
	[-0.0380880124867,-0.123980276287,0.799959719181]
])

robot_cloud = np.array([
	[0.0492373394284,-0.0195158571358,-0.149740619684],
	[0.0812812947064,-0.0162005132476,-0.149546702922],
	[0.129353613646,-0.00594445374758,-0.152700404264],
	[0.135790767295,-0.0428675653593,-0.152533638822],
	[0.146166739851,-0.0879996414536,-0.149734736876],
	[0.113056597944,-0.0921825044371,-0.153040620199],
	[0.107587085172,-0.0606446475798,-0.151996406645],
	[0.0573413696073,-0.063406048437,-0.151378853448],
	[0.0655033340241,-0.102204479661,-0.150394825623],
	[0.06559469181,-0.0895591581979,-0.118790831307],
	[0.0579372760734,-0.0419838738627,-0.115387724094],
	[0.0959611929873,-0.0573632132826,-0.119733208524],
	[0.137797322732,-0.0876853292552,-0.136078221224],
	[0.130026332678,-0.00868994416125,-0.127581110825],
	[0.0545468209731,-0.0240114526954,-0.0616319790133],
	[0.131028729506,-0.0753532076073,-0.0869359859178]
])


def solve_for_rigid_transformation(inpts, outpts):
    """
    Takes in two sets of corresponding points, returns the rigid transformation matrix from the first to the second.  """
    inpt_mean = inpts.mean(axis=0)
    outpt_mean = outpts.mean(axis=0)
    for i in range(outpts.shape[0]):
        outpts[i,:] -= outpt_mean
        inpts[i,:] -= inpt_mean
    X = inpts.T
    Y = outpts.T
    covariance = X.dot(Y.T)
    U, Sigma, V = np.linalg.svd(covariance)
    V = V.T
    idmatrix = np.identity(3)
    idmatrix[2, 2] = np.linalg.det(V.dot(U.T))
    R = V.dot(idmatrix).dot(U.T)
    t = outpt_mean.T - R.dot(inpt_mean.T)
    T = np.zeros((4,4))
    T[:3,:3] = R
    T[:3, -1] = t
    T[-1,-1] = 1
    return T


def test_predictions(camera_to_rob_mat, point_camera, point_robot_real, idx):
    """Test the predictions.
    """
    print('\n\tTEST CASE {}'.format(str(idx).zfill(2)))
    predicted_loc = cam_to_rob_mat.dot(point_camera)
    print("Actual:")
    print(point_robot_real[:-1])
    print("Predicted:")
    print(predicted_loc[:-1])
    print("Error (mm):")
    err_vec = 1000 * np.absolute(predicted_loc[:-1] - point_robot_real[:-1])
    print(err_vec)
    print("Total Dist (mm):")
    res = np.sum(err_vec**2)**.5
    print(res)
    return res


cam_to_rob_mat = solve_for_rigid_transformation(camera_cloud, robot_cloud)
print("The Camera/Robot Mat after calibration:")
print(cam_to_rob_mat)


goal_position = np.array([-0.0357646085322, -0.117930412292, 0.85547298193, 1.])

print(cam_to_rob_mat.dot(goal_position))
#
#
# # Daniel Seita: test case that Sam P added:
# camera_test = np.array([-0.0118521396071, -0.13668499887, 0.86562526226, 1])
# expected_loc =  np.array([0.105456516892, -0.0954783552311, -0.152453977653, 1])
# test_predictions(cam_to_rob_mat, camera_test, expected_loc, idx=0)
#
# # ------------------------------------------------------- #
# # Daniel Seita (Nov 21, 2019) doing some more test cases. #
# # ------------------------------------------------------- #
# camera_tests = [
#     np.array([ -0.00011895, -0.091267, 0.85667, 1]),
#     np.array([ -0.0079007,  -0.072769, 0.85832, 1]),
#     np.array([  0.015754,   -0.089749, 0.86588, 1]),
#     np.array([ -0.031009,   -0.077588, 0.86286, 1]),
#     np.array([  0.043873,   -0.12255,  0.86912, 1]),
# ]
# expected_locs = [
#     np.array([ 0.0877694352771, -0.0502514797531, -0.144702104175, 1]),
#     np.array([ 0.0911459528491, -0.0357664615125, -0.147040907758, 1]),
#     np.array([ 0.071191066603,  -0.0561317436764, -0.151678064479, 1]),
#     np.array([ 0.114796934298,  -0.0359131669317, -0.154084574282, 1]),
#     np.array([ 0.0537274230511, -0.0872453808799, -0.151112577282, 1]),
# ]
# errors = []
# for t, (point_camera, point_robot_real) in enumerate(zip(camera_tests, expected_locs)):
#     err = test_predictions(cam_to_rob_mat, point_camera, point_robot_real, idx=t+1)
#     errors.append(err)
# print('\nTotal errors: {},  {:.1f} +/- {:.1f}'.format(errors, np.mean(errors), np.std(errors)))
import math
a = math.acos(((0.36+0.6+0.6)-1)/2)*180./math.pi
print(a)