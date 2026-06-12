import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', os.urandom(24).hex())
    DEBUG = False
    TESTING = False

    PORT = int(os.getenv('PORT', 8080))
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    if os.getenv('RENDER'):
        UPLOAD_FOLDER = '/tmp/uploads'
        OUTPUT_FOLDER = '/tmp/outputs'
    else:
        UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
        OUTPUT_FOLDER = os.path.join(BASE_DIR, 'outputs')

    MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', 100 * 1024 * 1024))
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'geojson', 'json'}

    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*').split(',')

    USE_S3 = os.getenv('USE_S3', 'false').lower() == 'true'
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
    S3_REGION = os.getenv('S3_REGION', 'us-east-1')
    CLOUDINARY_URL = os.getenv('CLOUDINARY_URL')

    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL") or "sqlite:///s_tracer_backend.db"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        "pool_pre_ping": True,
        "pool_recycle": 280,
        "pool_size": 10,
        "max_overflow": 20,
    }

    CELL_SITE_MIN_SAMPLES = int(os.getenv('CELL_SITE_MIN_SAMPLES', 30))
    CELL_SITE_BIN_SIZE = int(os.getenv('CELL_SITE_BIN_SIZE', 5))

    @staticmethod
    def init_app():
        for folder in [Config.UPLOAD_FOLDER, Config.OUTPUT_FOLDER]:
            already_exists = os.path.isdir(folder)
            os.makedirs(folder, exist_ok=True)
            if not already_exists:
                print(f"Created directory: {folder}")


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False


class RenderConfig(ProductionConfig):
    pass


config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'render': RenderConfig,
    'default': DevelopmentConfig
}
