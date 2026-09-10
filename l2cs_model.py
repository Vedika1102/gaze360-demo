"""L2CS-Net gaze-estimation model (ResNet-50 backbone, binned yaw/pitch).

Matches the checkpoint layout published for the Gaze360-trained L2CS-Net:
keys are conv1/bn1/layer{1..4}/fc_yaw_gaze/fc_pitch_gaze/fc_finetune.

Reference: Abdelrahman et al., "L2CS-Net: Fine-Grained Gaze Estimation in
Unconstrained Environments" (2022). Trained on the Gaze360 dataset.
"""
import torch
import torch.nn as nn
from torchvision.models.resnet import Bottleneck, ResNet


class L2CS(ResNet):
    """ResNet-50 trunk with two binned-classification heads (yaw, pitch)."""

    def __init__(self, num_bins: int = 90):
        super().__init__(Bottleneck, [3, 4, 6, 3])
        # The trunk's original 1000-way classifier is unused; drop its params
        # so the checkpoint (which has no `fc.*`) loads with strict=True.
        self.fc = nn.Identity()

        feat_dim = 512 * Bottleneck.expansion  # 2048
        # fc_finetune exists in the checkpoint but is unused at inference time.
        self.fc_finetune = nn.Linear(feat_dim + 3, 3)
        self.fc_yaw_gaze = nn.Linear(feat_dim, num_bins)
        self.fc_pitch_gaze = nn.Linear(feat_dim, num_bins)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)          # AdaptiveAvgPool2d -> robust to input size
        x = torch.flatten(x, 1)

        return self.fc_yaw_gaze(x), self.fc_pitch_gaze(x)


def load_l2cs(weights_path: str, device: str = "cpu", num_bins: int = 90) -> L2CS:
    """Build L2CS and load a state-dict checkpoint (strict)."""
    model = L2CS(num_bins=num_bins)
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


class OnnxL2CS:
    """ONNX Runtime backend exposing the same (yaw_logits, pitch_logits) call.

    Returns torch tensors so downstream decoding/confidence code is unchanged.
    """

    def __init__(self, onnx_path: str, intra_threads: int = 0):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if intra_threads:
            so.intra_op_num_threads = intra_threads
        self.sess = ort.InferenceSession(onnx_path, sess_options=so,
                                         providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name

    def __call__(self, x):
        arr = x.detach().cpu().numpy() if hasattr(x, "detach") else x
        yaw, pitch = self.sess.run(None, {self.input_name: arr.astype("float32")})
        return torch.from_numpy(yaw), torch.from_numpy(pitch)
