"""Interpretable gaze x speaking fusion -> addressee state + robot role.

Every decision is traceable to named signals (no black box):
  - gaze cone angle to the camera/robot axis  (from Gaze360/L2CS)
  - gaze confidence                            (softmax entropy)
  - speaking                                   (visual ASD)

Per-person state: BYSTANDER | ATTENTIVE | ADDRESSING_ROBOT
Robot role:       PASSIVE   | ACTIVE (has a primary interlocutor)

Stage-1 limitation (documented): ADDRESSING_ROBOT = attentive AND speaking, so
a person who looks at the robot while talking to *other people* can misfire.
Disambiguating that needs multi-party mutual-gaze + turn-taking (VAP) — Stage 3.
"""
from collections import defaultdict

import numpy as np

from gaze_utils import angles_to_vector

CAMERA_AXIS = np.array([0.0, 0.0, -1.0])  # robot's view direction (our frontal gaze)


def gaze_cone_angle(yaw, pitch):
    """Angle (deg) between a person's gaze and the camera/robot axis."""
    v = angles_to_vector(yaw, pitch)
    v = v / (np.linalg.norm(v) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(np.dot(v, CAMERA_AXIS), -1, 1))))


class AddresseeFusion:
    def __init__(self, cone_deg=25.0, gaze_conf=0.5,
                 engage_frames=3, release_frames=8, speak_frames=5):
        self.cone_deg = cone_deg
        self.gaze_conf = gaze_conf
        self.engage_frames = engage_frames      # debounce: attentive must persist
        self.release_frames = release_frames     # ACTIVE holds through brief gaps
        self.speak_frames = speak_frames         # debounce: speaking must persist
        self._eng = defaultdict(int)             # consecutive attentive frames
        self._spk = defaultdict(int)             # consecutive speaking frames
        self._since_addressed = 0
        self.role = "PASSIVE"
        self.primary = None

    def update(self, per_person):
        """per_person: list of dicts with id, yaw, pitch, conf, speaking, speak_score.

        Returns (role, primary_id, states) where states[id] holds the full
        interpretable signal breakdown for logging/overlay.
        """
        states, addressing = {}, []
        seen = set()
        for p in per_person:
            pid = p["id"]
            seen.add(pid)
            angle = gaze_cone_angle(p["yaw"], p["pitch"])
            engaged = angle <= self.cone_deg and p["conf"] >= self.gaze_conf
            self._eng[pid] = self._eng[pid] + 1 if engaged else 0
            attentive = self._eng[pid] >= self.engage_frames

            self._spk[pid] = self._spk[pid] + 1 if p["speaking"] else 0
            speaking = self._spk[pid] >= self.speak_frames

            if attentive and speaking:
                st = "ADDRESSING_ROBOT"
                addressing.append(pid)
            elif attentive:
                st = "ATTENTIVE"
            else:
                st = "BYSTANDER"

            states[pid] = {
                "state": st, "cone_deg": round(angle, 1), "engaged": engaged,
                "attentive": attentive, "speaking": speaking,
                "speak_raw": bool(p["speaking"]),
                "speak_score": round(float(p["speak_score"]), 2),
                "conf": round(float(p["conf"]), 2),
            }

        # Robot role: engage the primary addresser; hold ACTIVE briefly through gaps.
        if addressing:
            self._since_addressed = 0
            if self.primary not in addressing:
                self.primary = addressing[0]
            self.role = "ACTIVE"
        else:
            self._since_addressed += 1
            if self._since_addressed > self.release_frames or self.primary not in seen:
                self.role = "PASSIVE"
                self.primary = None
        return self.role, self.primary, states
