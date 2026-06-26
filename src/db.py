from sqlalchemy import create_engine
from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASS

engine = create_engine(
    f"mysql+pymysql://{DB_USER}:{DB_PASS}@localhost/{DB_NAME}?unix_socket=/tmp/mysql.sock"
)