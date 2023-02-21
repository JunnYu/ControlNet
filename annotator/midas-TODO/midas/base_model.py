import paddle


class BaseModel(paddle.nn.Module):
    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        parameters = paddle.load(path, map_location="cpu")

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)
