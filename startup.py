import xgb_model as xgb
import sqlite3 as sql
import json
import sys
from ball_record import Ball


class MyObjectEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Ball):
            return obj.__dict__  # Or obj.__dict__ if suitable
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)





connectionString = "C://Users//Abhi//Documents//Schoolwork//Computer Science//ScorePredictor//PredictorUI//Database//CricketPredictorDB.db"

def getRecords(match_id):
    conn = sql.connect(connectionString)
    cur = sql.Cursor(conn)
    selectQuery = f"SELECT teamABatsmen, teamABowlers, teamBBatsmen, teamBBowlers from matchDetails where id = {match_id}"
    cur.execute(selectQuery)
    record = cur.fetchone()
    conn.close()
    if record:
        teamABatsmen = record[0]
        teamABowlers = record[1]
        teamBBatsmen = record[2]
        teamBBowlers = record[3] 
    else:
        print("No record found.")
    data1 = json.loads(teamABatsmen)
    data2 = json.loads(teamBBowlers)
    data3 = json.loads(teamBBatsmen)
    data4 = json.loads(teamABowlers)
    
    bl1, fis = xgb.simulate_innings(data1, data2, 1, 0)
    bl2, sis = xgb.simulate_innings(data3, data4, 2, fis)
    total_balls = bl2
    
    
    json_str = json.dumps(total_balls, indent=4, cls=MyObjectEncoder)
    updateQuery = f"UPDATE matchDetails set scoreCard = '{json_str}' where id = {match_id}"
    conn = sql.connect(connectionString)
    cur = sql.Cursor(conn)
    cur.execute(updateQuery)
    conn.commit()
    conn.close()
    print("success")


prm = int(sys.argv[1])
   
getRecords(prm)
