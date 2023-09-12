import os


class FileListener(object):

    def __init__(self, file_name, header):
        self.file_name = file_name
        self.header = header
        log_dir = "/".join(file_name.split("/")[:-1])
        if log_dir and not os.path.isdir(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        with open(file_name, "w") as file:
            file.write(";".join(header) + "\n")

    def listen(self, **kwargs):
        with open(self.file_name, "a") as file:
            file.write(";".join([str(kwargs.get(col, None)) for col in self.header]) + "\n")
