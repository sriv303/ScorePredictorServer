import xgb_model as xgb
import sqlite3 as sql
import json

connectionString = "C://Users//Abhi//Documents//Schoolwork//Computer Science//ScorePredictor//PredictorUI//Database//CricketPredictorDB.db"

def getRecords(match_id):
    conn = sql.connect(connectionString)
    cur = sql.Cursor(conn)
    selectQuery = f"SELECT teamABatsmen, teamABowlers, teamBBatsmen, teamBBowlers from matchDetails where id = {match_id}"
    cur.execute(selectQuery)
    record = cur.fetchone()
    if record:
        teamABatsmen = record[0]
        teamABowlers = record[1]
        teamBBatsmen = record[2]
        teamBBowlers = record[3] 
    else:
        print("No record found.")
    conn.close()
    data1 = json.loads(teamABatsmen)
    data2 = json.loads(teamBBowlers)
    data3 = json.loads(teamBBatsmen)
    data4 = json.loads(teamABowlers)
    
    bl1, fis = xgb.simulate_innings(data1, data2, 1, 0)
    bl2, sis = xgb.simulate_innings(data3, data4, 2, fis)
    
(getRecords(3))