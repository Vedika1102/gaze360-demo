"""Interpretable conversational controller: perception -> robot turn-taking action.

Consumes, each step, the addressee decision (who is talking to the robot) and
the user's vocal activity, and drives an inspectable state machine:

  robot state:  PASSIVE -> LISTENING -> SPEAKING (-> LISTENING | PASSIVE)
  actions:      IDLE, LISTEN, START_SPEAKING, CONTINUE, UTTERANCE_DONE,
                YIELD:RESUME, YIELD:ABANDON

Turn-taking while SPEAKING:
  - BACKCHANNEL ("mhm")     -> CONTINUE the current utterance (do not stop)
  - TURN_SHIFT (interrupt)  -> YIELD; RESUME if most of the utterance was already
                               delivered, else ABANDON (heuristic; a real system
                               would use NLU on the interruption content)

Every step returns a trace of the signals behind the decision (interpretable).
"""
from turntaking import BACKCHANNEL, TURN_SHIFT, HeuristicTurnTaking

PASSIVE, LISTENING, SPEAKING = "PASSIVE", "LISTENING", "SPEAKING"


class ConversationController:
    def __init__(self, utterance_s=4.0, resume_frac=0.6, lost_timeout_s=1.0,
                 turntaking=None):
        self.utterance_s = utterance_s
        self.resume_frac = resume_frac        # delivered fraction above which we RESUME
        self.lost_timeout = lost_timeout_s
        self.tt = turntaking or HeuristicTurnTaking()
        self.state = PASSIVE
        self._speak_start = None
        self._lost = 0.0
        self._t_prev = None

    def step(self, t, primary_addressing, primary_present, user_vocalizing):
        """primary_addressing: primary interlocutor is addressing the robot now.
        primary_present: the primary is still tracked. Returns (state, action, trace)."""
        dt = 0.0 if self._t_prev is None else max(t - self._t_prev, 0.0)
        self._t_prev = t
        robot_speaking = self.state == SPEAKING
        tt = self.tt.step(t, user_vocalizing, robot_speaking)
        ev = tt["event"]
        action = "IDLE"

        if self.state == PASSIVE:
            if primary_addressing:
                self.state = LISTENING
                action = "LISTEN"

        elif self.state == LISTENING:
            if not primary_present:
                self._lost += dt
                if self._lost >= self.lost_timeout:
                    self.state = PASSIVE
                    self._lost = 0.0
            else:
                self._lost = 0.0
                if ev == TURN_SHIFT:               # user finished -> robot takes floor
                    self.state = SPEAKING
                    self._speak_start = t
                    action = "START_SPEAKING"

        elif self.state == SPEAKING:
            delivered = (t - self._speak_start) / self.utterance_s
            if ev == BACKCHANNEL:
                action = "CONTINUE"
            elif ev == TURN_SHIFT:                 # interruption
                decision = "RESUME" if delivered >= self.resume_frac else "ABANDON"
                action = f"YIELD:{decision}"
                self.state = LISTENING
            elif delivered >= 1.0:
                action = "UTTERANCE_DONE"
                self.state = LISTENING
            else:
                action = "CONTINUE"

        trace = {"state": self.state, "action": action, "event": ev,
                 "tt_conf": tt["confidence"], "run_s": tt["run_s"],
                 "silence_s": tt["silence_s"],
                 "primary_addressing": primary_addressing,
                 "user_vocalizing": user_vocalizing}
        return self.state, action, trace
