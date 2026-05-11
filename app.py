from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import mysql.connector
from model_utils import predict_bot_response, train_cartpole_minimal, train_maze_minimal
app = Flask(__name__)
db_config = {'host': 'localhost', 'user': 'root', 'password': '', 'database': 'ai_project'}
def db_query(query, params=(), fetch=False):
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)
    cursor.execute(query, params)
    res = cursor.fetchall() if fetch else None
    conn.commit()
    cursor.close()
    conn.close()
    return res
@app.route('/')
def index():
    stats = db_query("SELECT * FROM agent_stats", fetch=True)
    return render_template('index.html', stats=stats)
@app.route('/train_view')
def train_view():
    mode = request.args.get("type")
    episodes = int(request.args.get("episodes", 50))

    def generate():
        try:
            yield f"data: [START] Обучение {mode}... \n\n"
            result_reward = train_cartpole_minimal(episodes) if mode == 'cartpole' else train_maze_minimal(episodes)
            db_query("""
                UPDATE agent_stats 
                SET total_episodes = total_episodes + %s, 
                    best_reward = GREATEST(best_reward, %s) 
                WHERE agent_name = %s
            """, (episodes, float(result_reward), mode))
            yield f"data: [SUCCESS] Рекорд сессии: {result_reward} pts. Статистика MySQL обновлена. \n\n"
        except Exception as e:
            yield f"data: [ERROR] Ошибка: {str(e)} \n\n"
        yield "data: [DONE] \n\n"
    return Response(stream_with_context(generate()), mimetype='text/event-stream')
if __name__ == '__main__':
    app.run(debug=False, threaded=True)