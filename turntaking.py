"""Turn-taking signal: classify a user's vocal activity into turn-taking events.

Interface (`TurnTakingSignal`) returns, each step, one of:
  HOLD        - nothing decisive
  BACKCHANNEL - brief vocalization during the robot's turn ("mhm", "yeah")
  TURN_SHIFT  - the floor should change (user finished their turn, OR user is
                grabbing the floor while the robot speaks = interruption)

`HeuristicTurnTaking` is a lightweight VAD+timing backend (no model): it keys
off vocalization run length, silence gaps, and whether the robot is speaking.
A pretrained VAP backend can later implement the same interface.
"""
HOLD, BACKCHANNEL, TURN_SHIFT = "HOLD", "BACKCHANNEL", "TURN_SHIFT"


class TurnTakingSignal:
    def step(self, t, user_vocalizing, robot_speaking):
        raise NotImplementedError


class HeuristicTurnTaking(TurnTakingSignal):
    def __init__(self, backchannel_max_s=0.6, turn_min_s=1.0, eot_gap_s=0.5):
        self.bc_max = backchannel_max_s      # <= this while robot speaks -> backchannel
        self.turn_min = turn_min_s           # >= this while robot speaks -> interruption
        self.eot_gap = eot_gap_s             # silence after user speech -> end of turn
        self._t_prev = None
        self._run = 0.0                      # current contiguous vocalization length
        self._silence = 0.0                  # silence since last vocalization
        self._spoke = False                  # user has vocalized in this segment
        self._fired = False                  # event already emitted for this segment

    def step(self, t, user_vocalizing, robot_speaking):
        dt = 0.0 if self._t_prev is None else max(t - self._t_prev, 0.0)
        self._t_prev = t

        if user_vocalizing:
            self._run += dt
            self._silence = 0.0
            self._spoke = True
        else:
            self._silence += dt

        event, conf = HOLD, 0.0

        if robot_speaking:
            if user_vocalizing:
                if self._run >= self.turn_min and not self._fired:
                    event, conf = TURN_SHIFT, min(self._run / self.turn_min, 1.0)  # interruption
                    self._fired = True
            else:
                # user's vocal burst ended during the robot's turn
                if self._spoke and 0 < self._run <= self.bc_max and not self._fired:
                    event, conf = BACKCHANNEL, 1.0 - self._run / self.bc_max
                if self._silence >= self.eot_gap:
                    self._reset()
        else:  # robot listening
            if self._spoke and not user_vocalizing and self._silence >= self.eot_gap \
                    and not self._fired:
                event, conf = TURN_SHIFT, 1.0     # user finished their turn
                self._fired = True
            if user_vocalizing and self._fired:
                self._reset()                     # user started a new turn

        return {"event": event, "confidence": round(conf, 2),
                "run_s": round(self._run, 2), "silence_s": round(self._silence, 2)}

    def _reset(self):
        self._run = 0.0
        self._silence = 0.0
        self._spoke = False
        self._fired = False
