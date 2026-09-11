"""Controlled quantitative eval of the addressee fusion / state machine.

Feeds the AddresseeFusion synthetic per-person signals (gaze yaw/pitch, gaze
confidence, speaking) for known multi-party scenarios and checks the output
per-person state and robot role against the expected labels. This isolates and
scores the fusion *logic* (no images/model), covering the cases that matter for
a robot: background speakers, low-confidence gaze, multi-party, engage/release.

    python eval_fusion.py
"""
import numpy as np

from addressee import AddresseeFusion

# archetypes: (yaw_rad, pitch_rad, conf, speaking)
AT_CAM = (0.05, -0.05, 0.85)       # gaze on the robot, high confidence
OFF_AXIS = (1.0, 0.0, 0.85)        # ~57 deg off-axis
LOW_CONF = (0.05, -0.05, 0.35)     # on-axis but unconfident


def person(pid, arche, speaking):
    y, p, c = arche
    return {"id": pid, "yaw": y, "pitch": p, "conf": c,
            "speaking": speaking, "speak_score": 9.0 if speaking else 0.0}


# scenario = (name, people_per_frame, expected_states{id:state}, exp_role, known_gap)
SCENARIOS = [
    ("solo addresser",
     [person(0, AT_CAM, True)], {0: "ADDRESSING_ROBOT"}, "ACTIVE", False),
    ("attentive listener only",
     [person(0, AT_CAM, False)], {0: "ATTENTIVE"}, "PASSIVE", False),
    ("addresser + background speaker",
     [person(0, AT_CAM, True), person(1, OFF_AXIS, True)],
     {0: "ADDRESSING_ROBOT", 1: "BYSTANDER"}, "ACTIVE", False),
    ("addresser + attentive listener",
     [person(0, AT_CAM, True), person(1, AT_CAM, False)],
     {0: "ADDRESSING_ROBOT", 1: "ATTENTIVE"}, "ACTIVE", False),
    ("only background chatter",
     [person(0, OFF_AXIS, True), person(1, OFF_AXIS, True)],
     {0: "BYSTANDER", 1: "BYSTANDER"}, "PASSIVE", False),
    ("low-confidence gaze",
     [person(0, LOW_CONF, True)], {0: "BYSTANDER"}, "PASSIVE", False),
    # KNOWN Stage-1 gap: looking at robot but talking to others -> misfires.
    ("looking-but-addressing-others (known gap)",
     [person(0, AT_CAM, True)], {0: "ATTENTIVE"}, "PASSIVE", True),
]


def run_scenario(people, frames=10):
    """Run the fusion for N steady frames; return final (role, states)."""
    fusion = AddresseeFusion()
    role = primary = None
    states = {}
    for _ in range(frames):
        role, primary, states = fusion.update(people)
    return role, primary, states


def main():
    total = correct = 0
    role_ok = role_total = 0
    print(f"{'scenario':42} {'result':10} detail")
    for name, people, exp_states, exp_role, known in SCENARIOS:
        role, primary, states = run_scenario(people)
        per_ok = all(states[i]["state"] == exp_states[i] for i in exp_states)
        r_ok = role == exp_role
        if not known:
            for i in exp_states:
                total += 1
                correct += int(states[i]["state"] == exp_states[i])
            role_total += 1
            role_ok += int(r_ok)
        tag = "KNOWN-GAP" if known else ("PASS" if (per_ok and r_ok) else "FAIL")
        got = {i: states[i]["state"] for i in exp_states}
        print(f"{name:42} {tag:10} role={role} states={got}")

    print(f"\nper-person state accuracy: {correct}/{total} = {correct/total:.2%}")
    print(f"robot-role accuracy:       {role_ok}/{role_total} = {role_ok/role_total:.2%}")

    # temporal: engage then disengage -> ACTIVE then PASSIVE after release
    fusion = AddresseeFusion(release_frames=5)
    for _ in range(6):
        role, _, _ = fusion.update([person(0, AT_CAM, True)])
    engaged_role = role
    for _ in range(7):
        role, _, _ = fusion.update([person(0, AT_CAM, False)])  # stops speaking+looking? still attentive
    for _ in range(7):
        role, _, _ = fusion.update([])  # person leaves
    print(f"\ntemporal: engaged->{engaged_role}, after leave+release->{role} "
          f"({'PASS' if engaged_role=='ACTIVE' and role=='PASSIVE' else 'FAIL'})")


if __name__ == "__main__":
    main()
