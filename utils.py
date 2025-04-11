import pandas as pd
import numpy as np

FOOTBALL_FIELD_LENGTH = 105
FOOTBALL_FIELD_WIDTH = 68
MAX_X = 5830.00
MAX_Y = 4020.00
MIN_X = -5830.00
MIN_Y = -4020.00

def get_player_with_ball(series, threshold=83.29):
    ball_x = series['ball_x']
    ball_y = series['ball_y']
    
    min_distance = float('inf')
    player_with_ball = None
    player_names = ["_".join(col.split('_')[:-1]) for col in series.index if (col.startswith('home') or col.startswith('away')) and col.endswith('_x')]
    
    for player_name in player_names:
        x_col = f'{player_name}_x'
        y_col = f'{player_name}_y'
        
        distance = np.sqrt(
            (series[x_col] - ball_x) ** 2 + 
            (series[y_col] - ball_y) ** 2
        )
        if distance < min_distance and distance < threshold:
            min_distance = distance
            player_with_ball = player_name
 
    return player_with_ball

def get_ball_possession(df):
    df['ball_possession'] = df.apply(get_player_with_ball, axis=1)
    df['ball_possession'] = df['ball_possession'].ffill().bfill()
    return df

def get_pass_history(df):
    pass_history= {}
    df['player_with_ball'] = df.apply(get_player_with_ball, axis=1)
    non_values = df[df['player_with_ball'].notna()]['player_with_ball']
    non_dup_values =[non_values.iat[0]] + [non_values.iat[val] for val in range(1,len(non_values)) if non_values.iat[val] !=non_values.iat[val-1]]
    
    for player_dix in range(len(non_dup_values)-1):
        if non_dup_values[player_dix] not in pass_history:
            pass_history[non_dup_values[player_dix]] = [non_dup_values[player_dix+1]]
        else:
            pass_history[non_dup_values[player_dix]].append(non_dup_values[player_dix+1])
    return pass_history

def get_pass_stats(pass_history):
    pass_stats={}
    for player, passes in pass_history.items():
        player_team=player.split('_')[0]
        valid_passes=[pass_ for pass_ in passes if player_team in pass_]
        pass_stats[player]={"valid_passes": len(valid_passes), "missed_passes": len(passes)-len(valid_passes), "pass_accuracy": len(valid_passes)/len(passes)}
    return pass_stats

def convert_relative_coordinates_to_real_coordinates(x, y):
    x_real = x * FOOTBALL_FIELD_LENGTH / MAX_X
    y_real = y * FOOTBALL_FIELD_WIDTH / MAX_Y
    return x_real, y_real

def convert_real_coordinates_to_relative_coordinates(x, y):
    x_relative = x * MAX_X / FOOTBALL_FIELD_LENGTH
    y_relative = y * MAX_Y / FOOTBALL_FIELD_WIDTH
    return x_relative, y_relative