from random import randint
import numpy as np

class BaseballGame:
    
    def __init__(self, pitch_model, event_model, options = {}) -> None:
        self.pitch_model = pitch_model
        self.event_model = event_model
        self.state = {
            "b_score": 0,
            "p_score": 0,
            "b_count": 0,
            "s_count": 0,
            "outs": 0,
            "pitch_num": 1,
            "on_1b": 0,
            "on_2b": 0,
            "on_3b": 0,
            "inning": 1,
            "p_isrighty": 1,
            "b_isrighty": 0,
            "is_top_inning": 1
        }

        self.options = {
            "MAX_RUNS": 50
        }

        if "MAX_RUNS" in options: self.options["MAX_RUNS"] = options["MAX_RUNS"]

    def run_sim(self, n_games = 1) -> float:
        """
        Simulates many baseball games and returns average runs

        :param int n_games: Number of games to simulate and average over
        :return: Average runs scored by model
        """

        runs = np.empty(n_games)
        for i in range(n_games):
            # total_runs += self.game()
            runs[i] = self.game()/2
        
        print(f"Average: {np.average(runs)}, STD: {np.std(runs)}")
        return runs

    def game(self) -> int:
        self.init_state()

        while not self.is_game_over():
            self.inning()

        return self.state["b_score"] + self.state["p_score"]

    def inning(self) -> None:
        self.state["outs"] = 0
        self.state["on_1b"], self.state["on_1b"], self.state["on_1b"] = 0, 0, 0
        self.state["is_top_inning"] = 1

        while not self.is_inning_over():
            # print(self.state)
            self.at_bat()

            if self.state["outs"] >= 3 and self.state["is_top_inning"] == 1:
                self.state["outs"] = 0
                self.state["on_1b"], self.state["on_1b"], self.state["on_1b"] = 0, 0, 0
                self.state["is_top_inning"] = 0

                self.swap_scores()
    
        self.swap_scores()
        self.state["inning"] += 1

    def at_bat(self) -> None:
        self.state["b_count"], self.state["s_count"] = 0, 0
        self.state["pitch_num"] = 1
        self.state["p_isrighty"] = randint(0, 1)
        self.state["b_isrighty"] = randint(0, 1)
        
        at_bat_over = False
        while not at_bat_over:
            pitch = self.pitch_model.throw_pitch(self.state)
            result = self.event_model.det_event(pitch, self.state)
            at_bat_over = self.sim_event(result)
            # print(result, self.state)

    def sim_event(self, event) -> bool:
        self.state["pitch_num"] += 1
        match event:
            case "Ball":
                return self.ball()
            case "Foul Ball":
                if self.state["b_count"] < 2: self.state["b_count"] += 1
                return False
            case "Called Strike":
                return self.strike()
            case "Swinging Strike":
                return self.strike()
            case "Groundout":
                self.state["outs"] += 1
                return True
            case "Single":
                self.state["b_score"] += self.state["on_3b"]
                self.state["on_3b"] = self.state["on_2b"]
                self.state["on_2b"] = self.state["on_1b"]
                self.state["on_1b"] = 1
                return True
            case "Flyout":
                self.state["outs"] += 1
                return True
            case "Ball in Dirt":
                return self.ball()
            case "Lineout":
                self.state["outs"] += 1
                return True
            case "Pop Out":
                self.state["outs"] += 1
                return True
            case "Double":
                self.state["b_score"] += self.state["on_2b"] + self.state["on_3b"]
                self.state["on_3b"] = self.state["on_1b"]
                self.state["on_2b"] = 1
                self.state["on_1b"] = 0
                return True
            case "Foul Tip":
                return self.strike()
            case "Swinging Strike (Blocked)":
                return self.strike()
            case "Home Run":
                self.state["b_score"] += 1 + self.state["on_1b"] + self.state["on_2b"] + self.state["on_3b"]
                return True
            case "Forceout":
                self.state["outs"] += 1
                return True
            case "Grounded Into DP":
                self.state["outs"] += 2
                
                if self.state["on_1b"] == 1:
                    self.state["on_1b"] = 0
                else:
                    self.state["on_2b"] = 0
                    self.state["on_1b"] = 0
                return True
            case "Foul Bunt":
                if self.state["b_count"] < 2: self.state["b_count"] += 1
                return False
            case "Sac Fly":
                self.state["outs"] += 1
                self.state["b_score"] += self.state["on_3b"]
                self.state["on_3b"] = self.state["on_2b"]
                self.state["on_2b"] = self.state["on_1b"]
                self.state["on_1b"] = 0
                return True
            case "Sac Bunt":
                self.state["outs"] += 1
                self.state["b_score"] += self.state["on_3b"]
                self.state["on_3b"] = self.state["on_2b"]
                self.state["on_2b"] = self.state["on_1b"]
                self.state["on_1b"] = 0
                return True
            case "Triple":
                self.state["b_score"] += self.state["on_1b"] + self.state["on_2b"] + self.state["on_3b"]
                self.state["on_3b"] = 1
                self.state["on_2b"] = 0
                self.state["on_1b"] = 0
            case "Hit by pitch":
                self.walk()
                return True
            case "Missed Bunt":
                return self.strike()
            case "Double Play":
                self.state["outs"] += 2
                
                if self.state["on_1b"] == 1:
                    self.state["on_1b"] = 0
                else:
                    self.state["on_2b"] = 0
                    self.state["on_1b"] = 0
                return True
            case "Bunt Groundout":
                self.state["outs"] += 1
                return True
            case "Bunt Pop Out":
                self.state["outs"] += 1
                return True
            case "Sac Fly DP":
                self.state["outs"] += 2
                
                if self.state["on_1b"] == 1:
                    self.state["on_1b"] = 0
                else:
                    self.state["on_2b"] = 0
                    self.state["on_1b"] = 0
                return True
            case "Triple Play":
                self.state["outs"] += 3
                return True
        
        raise Exception("Did not return event") 
   
    def is_game_over(self) -> bool:
        if (self.state["b_score"] + self.state["p_score"]) > self.options["MAX_RUNS"]:
            return True

        if self.state["inning"] > 9 and (self.state["b_score"] != self.state["p_score"]):
            return True

        return False

    def is_inning_over(self) -> bool:
        if (self.state["b_score"] + self.state["p_score"]) > self.options["MAX_RUNS"]:
            return True

        if self.state["is_top_inning"] == 0 and self.state["outs"] >= 3:
            return True

        return False

    def ball(self) -> bool:
        self.state["b_count"] += 1

        if self.state["b_count"] == 4:
            self.walk()
            return True
        else:
            return False

    def strike(self) -> bool:
        self.state["s_count"] += 1

        if self.state["s_count"] == 3:
            self.state["outs"] += 1
            return True
        else:
            return False

    def walk(self) -> None:
        self.state["b_score"] += self.state["on_1b"] and self.state["on_2b"] and self.state["on_3b"]
        self.state["on_1b"] = 1
        self.state["on_2b"] = self.state["on_2b"] or self.state["on_1b"]
        self.state["on_3b"] = self.state["on_3b"] or (self.state["on_1b"] and self.state["on_2b"])
                    
    def swap_scores(self):
        temp = self.state["b_score"]
        self.state["b_score"] = self.state["p_score"]
        self.state["p_score"] = temp

    def init_state(self) -> None:
        self.state = {
            "b_score": 0,
            "p_score": 0,
            "b_count": 0,
            "s_count": 0,
            "outs": 0,
            "pitch_num": 1,
            "on_1b": 0,
            "on_2b": 0,
            "on_3b": 0,
            "inning": 1,
            "p_isrighty": 1,
            "b_isrighty": 0,
            "is_top_inning": 1
        }