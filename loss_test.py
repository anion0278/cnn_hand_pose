
import sklearn.metrics as sm
y_true = [1, 1, 1, 1, 1]
y_pred = [0,0,0,0.4,0]
print("Mean abs: %s" % sm.mean_absolute_error(y_true, y_pred))
print("Mean sqr: %s" % sm.mean_squared_error(y_true, y_pred))
print("Mean log: %s" % sm.mean_squared_log_error(y_true, y_pred))
print("Hinge: %s" % sm.hinge_loss(y_true, y_pred))
print("Log cosh: %s" % sm.log_loss(y_true, y_pred))
