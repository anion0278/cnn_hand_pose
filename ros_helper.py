#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

ros_packages = "/opt/ros/kinetic/lib/python2.7/dist-packages"
if (ros_packages in sys.path):
	print("Removed invalid ros package path")
	sys.path.remove(ros_packages)

print("Python %s" % sys.version_info[0])
