import math


def moving_weight(current_epoch, total_epochs, weight_begin=0.95, weight_end=0.99):
    tao = weight_end - (weight_end - weight_begin) * (math.cos(math.pi * min(1, current_epoch / total_epochs)) + 1) / 2
    return tao


def exponential_moving_average(teacher, student, current_epoch, total_epochs, tao_begin=0.95, tao_end=0.99):
    tao = moving_weight(current_epoch, total_epochs, tao_begin, tao_end)
    teacher_param = teacher.state_dict()
    student_param = student.state_dict()
    for key in teacher_param.keys():
        teacher_param[key] = tao * teacher_param[key] + (1 - tao) * student_param[key]
    teacher.load_state_dict(teacher_param)


def bn_statistics_moving_average(source_bn_statistics, target_bn_statistics, current_epoch, total_epochs,
                                 tao_begin=0.95, tao_end=0.99):
    tao = moving_weight(current_epoch, total_epochs, tao_begin, tao_end)
    for key in source_bn_statistics.keys():
        source_bn_statistics[key] = tao * source_bn_statistics[key] + (1 - tao) * target_bn_statistics[key]


def cotta_ema(teacher, student, alpha=0.999):
    for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
        teacher_param.data[:] = alpha * teacher_param[:].data[:] + (1 - alpha) * student_param[:].data[:]
    return teacher
