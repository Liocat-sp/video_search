import psycopg2
import datetime
import json

# Now connect to the new database
conn = psycopg2.connect(
    dbname='video_search',
    user='postgres',
    password='password',
    host='localhost',
    port='5432'
)

cursor = conn.cursor()



def create_video_frames(frame_number, data, embedding):
    timestamp = datetime.datetime.now()

    if hasattr(data, 'tolist'):
        json_data = json.dumps(data.tolist())
    else:
        json_data = json.dumps(data)
    
    cursor.execute("""    
        INSERT INTO video_frames 
        (frame_number, object_class, frame_data, embedding, timestamp)
        VALUES (%s, %s, %s, %s, %s)
    """, (frame_number, '', json_data, embedding, timestamp)
    )

    conn.commit()



def search_video_frame(embedding):
    cursor.execute(
        """
            SELECT * FROM video_frames 
                WHERE embedding <-> %s::vector > 0.6 
                ORDER BY embedding <-> %s::vector LIMIT 5
        """,
        (embedding, embedding)
    )
    result = cursor.fetchall()
    return result

def delete_videos():
    cursor.execute("DELETE FROM video_frames;")
    conn.commit();