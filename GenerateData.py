import glob
import warnings
import pandas as pd

from utils import *
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--param_path", dest = "path", help = "path to parameter json file", default = None)
args = parser.parse_args()
params = load_json(args.path)

class Create_Data():
    def __init__(self, csv):
        self.df = pd.read_csv(csv, index_col = False, encoding = 'utf-8')
        self.df_raw = None
        self.player_name = []
        self.team_name = None
        self.cheer_word = params["cheer_words"]
        self.tactic_words = params["tactic_words"]
        self.coach_words = params["coach_words"]
        self.stream_words = params["stream_words"]
        self.judge_words = params["judge_words"]
        self.cheer_emoji = params["cheer_emoji"]
        self.team_emoji = params["team_emoji"]

    def Raw_df(self):
        df_ = self.df.copy()
        self.df_raw = Preprocess(df_)
        self.df_raw.to_csv("data/RawData.csv", index = False)
        return self

    def TeamAndPlayer(self):
        file = glob.glob('data/player/*.txt')

        for t in file:
            with open(t, 'r') as f:
                names = [name.strip() for name in f.readlines()]
                self.player_name.extend(names)

        team_file = 'data/team/team.txt'
        with open(team_file, 'r') as f:
            self.team_name = [t.strip() for t in f.readlines()]

        self.team_name.append('球隊')
        return self


def main():
    warnings.filterwarnings("ignore")
    csv = "data/chat.csv"
    data = Create_Data(csv)
    data.Raw_df().TeamAndPlayer()
    General_df = data.df_raw
    cheer, game, broad, chat = [], [], [], []
    player, coach, judge, tactic, team = [], [], [], [], []

    for index, row in General_df.iterrows():

        ## Cheer
        if (Contain_KeyWord(row["text"], data.cheer_word)) or (Contain_KeyWord(row["text"], data.cheer_emoji)):
            cheer.append(1)
        else:
            cheer.append(0)
        
        ## Broadcast
        if Contain_KeyWord(row["text"], data.stream_words):
            broad.append(1)
        else:
            broad.append(0)
        
        ## Game
        if (Contain_KeyWord(row["text"], data.tactic_words)) or \
           (Contain_KeyWord(row["text"], data.coach_words)) or \
           (Contain_KeyWord(row["text"], data.judge_words)) or \
           (Contain_KeyWord(row["text"], data.team_emoji)) or \
           (Contain_KeyWord(row["text"], data.team_name)) or \
           (Contain_KeyWord(row["text"], data.player_name)):
            game.append(1)
        else:
            game.append(0)
        
        ## Chat
        if cheer[-1] == 0 and broad[-1] == 0 and game[-1] == 0:
            chat.append(1)
        else:
            chat.append(0)

    General_df["Cheer"] = cheer
    General_df["Game"]  = game
    General_df["Broadcast"] = broad
    General_df["Chat"] = chat
    General_df.to_csv("data/GeneralLayer.csv", index = False)

    Game_df = General_df[General_df["Game"] == 1].drop(columns = ["Cheer", "Game", "Broadcast", "Chat"])
    for index, row in Game_df.iterrows():

        ## Player
        if Contain_KeyWord(row["text"], data.player_name):
            player.append(1)
        else:
            player.append(0)
        
        ## Coach
        if Contain_KeyWord(row["text"], data.coach_words):
            coach.append(1)
        else:
            coach.append(0)
        
        ## Judge
        if Contain_KeyWord(row["text"], data.judge_words):
            judge.append(1)
        else:
            judge.append(0)
        
        ## Tactic
        if Contain_KeyWord(row["text"], data.tactic_words):
            tactic.append(1)
        else:
            tactic.append(0)
        
        ## Team
        if (Contain_KeyWord(row["text"], data.team_emoji)) or (Contain_KeyWord(row["text"], data.team_name)):
            team.append(1)
        else:
            team.append(0)
    
    Game_df["Player"] = player
    Game_df["Coach"] = coach
    Game_df["Judge"] = judge
    Game_df["Tactic"] = tactic
    Game_df["Team"] = team
    Game_df.to_csv("data/GameLayer.csv", index = False)

if __name__ == "__main__":
    main()