# -*- coding: utf-8 -*-

import zmq

class TcpServer:
    def __init__(self, port, callback_processor):
        self.port = port
        self.callback = callback_processor

    def start_server_mode_sync(self, model):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind("tcp://*:%s" % self.port)
        print("Server started on port:", self.port)
        server_command = "PredictImage:"
        while (True):
            responce = self.callback(model, request = socket.recv_string())
            socket.send_string(str(responce))