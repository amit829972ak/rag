import os
import datetime
import logging
import json
import time
import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.pool import NullPool
from sqlalchemy import (
    create_engine, Column, Integer, String, 
    ForeignKey, DateTime, Text, LargeBinary
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Base class for SQLAlchemy models
Base = declarative_base()

# Define models
class User(Base):
    """User model to store user information."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    username = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, username={self.username})>"

class Conversation(Base):
    """Conversation model to group related messages."""
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    title = Column(String(255), default="New Conversation")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Conversation(id={self.id}, user_id={self.user_id}, title={self.title})>"

class Message(Base):
    """Message model to store chat messages."""
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"))
    role = Column(String(50))  # 'user' or 'assistant'
    content = Column(Text)
    image_data = Column(LargeBinary, nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
    conversation = relationship("Conversation", back_populates="messages")
    
    def __repr__(self):
        return f"<Message(id={self.id}, conversation_id={self.conversation_id}, role={self.role})>"

class KnowledgeItem(Base):
    """Model for storing knowledge items with embeddings."""
    __tablename__ = "knowledge_items"
    
    id = Column(Integer, primary_key=True)
    content = Column(Text)
    embedding = Column(Text)  # JSON serialized embedding
    source = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    def __repr__(self):
        return f"<KnowledgeItem(id={self.id}, source={self.source})>"

# Database connection singleton
ENGINE = None
SESSION = None

def get_database_url():
    """Get the database URL from environment variables or use SQLite as fallback."""
    db_url = os.environ.get("DATABASE_URL")
    
    if db_url:
        # Adapt the URL if it's a Postgres URL from Heroku
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
        return db_url
    else:
        # Use SQLite as a fallback
        sqlite_path = "sqlite:///chatbot.db"
        logger.warning(f"No DATABASE_URL found, using SQLite: {sqlite_path}")
        return sqlite_path

def initialize_database():
    """Create database tables if they don't exist with retry logic."""
    global ENGINE, SESSION
    
    # If already initialized, return
    if ENGINE is not None and SESSION is not None:
        return
    
    db_url = get_database_url()
    
    try:
        # Create engine with appropriate connection settings
        if db_url.startswith("postgresql"):
            ENGINE = create_engine(
                db_url,
                echo=False,
                pool_pre_ping=True,
                connect_args={
                    "connect_timeout": 10,
                }
            )
        else:
            # SQLite doesn't need pooling or similar settings
            ENGINE = create_engine(db_url, echo=False)
        
        # Create a session maker
        SESSION = sessionmaker(bind=ENGINE)
        
        # Create tables
        Base.metadata.create_all(ENGINE)
        logger.info("Database tables created successfully")
        print("Database tables created successfully")
        
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        print(f"Error initializing database: {str(e)}")
        
        # If using PostgreSQL, try SQLite as fallback
        if db_url.startswith("postgresql"):
            logger.warning("Falling back to SQLite database")
            sqlite_path = "sqlite:///chatbot.db"
            
            try:
                ENGINE = create_engine(sqlite_path, echo=False)
                SESSION = sessionmaker(bind=ENGINE)
                Base.metadata.create_all(ENGINE)
                logger.info("Created SQLite database as fallback")
            except Exception as sqlite_e:
                logger.error(f"Error creating SQLite fallback: {str(sqlite_e)}")
                raise

def execute_with_retry(func, *args, **kwargs):
    """Execute a database function with retry logic."""
    max_retries = 3
    retry_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Database error: {str(e)}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                logger.error(f"Database operation failed after {max_retries} attempts: {str(e)}")
                raise

def get_or_create_user(username=None):
    """Get or create a user."""
    def _get_or_create_user():
        if SESSION is None:
            initialize_database()
            
        session = SESSION()
        try:
            # Look for an existing user or create a new one
            user = session.query(User).first()
            if not user:
                user = User(username=username)
                session.add(user)
                session.commit()
            return user
        finally:
            session.close()
    
    return execute_with_retry(_get_or_create_user)

def get_or_create_conversation(user_id, title=None):
    """Get or create a conversation for a user."""
    def _get_or_create_conversation():
        if SESSION is None:
            initialize_database()
            
        session = SESSION()
        try:
            # Look for an existing conversation or create a new one
            conversation = (
                session.query(Conversation)
                .filter(Conversation.user_id == user_id)
                .order_by(Conversation.created_at.desc())
                .first()
            )
            
            if not conversation or title == "New Conversation":
                conversation = Conversation(
                    user_id=user_id,
                    title=title or "New Conversation"
                )
                session.add(conversation)
                session.commit()
                
            return conversation
        finally:
            session.close()
    
    return execute_with_retry(_get_or_create_conversation)

def add_message_to_db(conversation_id, role, content, image_data=None):
    """Add a message to the database."""
    def _add_message():
        if SESSION is None:
            initialize_database()
            
        session = SESSION()
        try:
            message = Message(
                conversation_id=conversation_id,
                role=role,
                content=content,
                image_data=image_data,
                timestamp=datetime.datetime.utcnow()
            )
            session.add(message)
            session.commit()
            return message
        finally:
            session.close()
    
    return execute_with_retry(_add_message)

def get_conversation_messages(conversation_id, limit=100):
    """Get messages for a conversation."""
    def _get_messages():
        if SESSION is None:
            initialize_database()
            
        session = SESSION()
        try:
            messages = (
                session.query(Message)
                .filter(Message.conversation_id == conversation_id)
                .order_by(Message.timestamp)
                .limit(limit)
                .all()
            )
            return messages
        finally:
            session.close()
    
    return execute_with_retry(_get_messages)

def add_knowledge_item(content, embedding, source=None):
    """Add a knowledge item with embedding to the database."""
    def _add_knowledge_item():
        if SESSION is None:
            initialize_database()
            
        session = SESSION()
        try:
            # Serialize the embedding to JSON
            embedding_json = json.dumps(embedding)
            
            item = KnowledgeItem(
                content=content,
                embedding=embedding_json,
                source=source,
                created_at=datetime.datetime.utcnow()
            )
            session.add(item)
            session.commit()
            return item
        finally:
            session.close()
    
    return execute_with_retry(_add_knowledge_item)

def get_all_knowledge_items():
    """Get all knowledge items with embeddings."""
    def _get_items():
        if SESSION is None:
            initialize_database()
            
        session = SESSION()
        try:
            items = session.query(KnowledgeItem).all()
            
            # Parse the embeddings from JSON
            result = []
            for item in items:
                try:
                    embedding = json.loads(item.embedding)
                    result.append({
                        'id': item.id,
                        'content': item.content,
                        'embedding': embedding,
                        'source': item.source
                    })
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"Invalid embedding for item {item.id}, skipping")
                    
            return result
        finally:
            session.close()
    
    return execute_with_retry(_get_items)
