"""Scenario test for the conversational controller (turn-taking policy).

Drives the controller with scripted observation timelines (user vocal activity +
addressee state at 25 fps) and checks the resulting action/state sequence for
the key conversational behaviors: engage, robot-takes-turn, backchannel
(continue), early interruption (abandon), late interruption (resume), disengage.
"""
from controller import ConversationController, PASSIVE, SPEAKING

FPS = 25.0
DT = 1.0 / FPS


def run(segments, utterance_s=4.0):
    """segments: list of (dur_s, user_vocalizing, addressing, present).
    Returns (notable_events, saw_backchannel, final_state)."""
    c = ConversationController(utterance_s=utterance_s)
    t = 0.0
    notable, saw_bc = [], False
    for dur, voc, addr, pres in segments:
        for _ in range(int(round(dur * FPS))):
            state, action, tr = c.step(t, addr, pres, voc)
            if tr["event"] == "BACKCHANNEL":
                saw_bc = True
            if action not in ("IDLE", "CONTINUE") and (not notable or notable[-1] != action):
                notable.append(action)
            t += DT
    return notable, saw_bc, c.state


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}   {detail}")
    return cond


def main():
    results = []

    # 1. Engage + robot takes the turn after user finishes.
    ev, _, st = run([(2.0, True, True, True), (1.0, False, False, True)])
    results.append(check("engage->take-turn", "LISTEN" in ev and "START_SPEAKING" in ev,
                         f"events={ev}"))

    # 2. Backchannel during robot's turn -> CONTINUE (no yield).
    ev, bc, st = run([(2.0, True, True, True), (0.8, False, False, True),   # engage->speak
                      (0.3, True, False, True),                              # brief "mhm"
                      (1.0, False, False, True)])
    results.append(check("backchannel->continue",
                         bc and not any("YIELD" in e for e in ev) and st == SPEAKING,
                         f"saw_bc={bc} state={st} events={ev}"))

    # 3. Early interruption -> YIELD:ABANDON.
    ev, _, st = run([(2.0, True, True, True), (0.8, False, False, True),    # engage->speak
                     (1.4, True, False, True)])                             # sustained early
    results.append(check("early-interrupt->abandon", "YIELD:ABANDON" in ev, f"events={ev}"))

    # 4. Late interruption -> YIELD:RESUME (most of a longer utterance delivered).
    ev, _, st = run([(2.0, True, True, True), (0.8, False, False, True),    # engage->speak
                     (4.0, False, False, True),                             # robot speaks alone
                     (1.4, True, False, True)],                             # interrupt late
                    utterance_s=6.0)
    results.append(check("late-interrupt->resume", "YIELD:RESUME" in ev, f"events={ev}"))

    # 5. Disengage -> PASSIVE.
    ev, _, st = run([(1.0, True, True, True), (1.5, False, False, False)])
    results.append(check("disengage->passive", st == PASSIVE, f"final_state={st}"))

    print(f"\ncontroller scenarios passed: {sum(results)}/{len(results)}")


if __name__ == "__main__":
    main()
