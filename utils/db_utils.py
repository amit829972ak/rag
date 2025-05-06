import os
import json
import datetime
import base64
import time
import logging
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, LargeBinary, ForeignKey, Boolean, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, scoped_session
from sqlalchemy.pool import QueuePool

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get database URL from environment variables
DATABASE_URL = os.getenv("DATABASE_URL")

# Create SQLAlchemy engine with connection pooling and retry logic
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800,  # Recycle connections after 30 minutes
    connect_args={
        "connect_timeout": 10,
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 10,
        "keepalives_count": 5
    }
)

# Create scoped session for thread safety
session_factory = sessionmaker(bind=engine)
Session = scoped_session(session_factory)

# Create base class for models
Base = declarative_base()

class User(Base):
    """User model to store user information."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    username = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationships
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
    
    # Relationships
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Conversation(id={self.id}, title={self.title})>"


class Message(Base):
    """Message model to store chat messages."""
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"))
    role = Column(String(50))  # 'user' or 'assistant'
    content = Column(Text)
    image_data = Column(LargeBinary, nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    
    def __repr__(self):
        return f"<Message(id={self.id}, role={self.role})>"


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


def initialize_database():
    """Create database tables if they don't exist with retry logic."""
    max_retries = 5
    retry_count = 0
    backoff_factor = 1.5
    
    while retry_count < max_retries:
        try:
            # Test connection first
            conn = engine.connect()
            conn.close()
            
            # Create tables
            Base.metadata.create_all(engine)
            logger.info("Database tables created successfully")
            print("Database tables created successfully")
            return True
        except Exception as e:
            retry_count += 1
            wait_time = backoff_factor ** retry_count
            logger.error(f"Database connection error: {str(e)}. Retrying in {wait_time:.2f} seconds (Attempt {retry_count}/{max_retries})")
            
            if retry_count >= max_retries:
                logger.error(f"Failed to initialize database after {max_retries} attempts: {str(e)}")
                print(f"Failed to initialize database: {str(e)}")
                return False
            
            time.sleep(wait_time)
    
    return False


def execute_with_retry(func, *args, **kwargs):
    """Execute a database function with retry logic."""
    max_retries = 3
    retry_count = 0
    backoff_factor = 1.5
    
    while retry_count < max_retries:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            retry_count += 1
            wait_time = backoff_factor ** retry_count
            logger.error(f"Database operation error: {str(e)}. Retrying in {wait_time:.2f} seconds (Attempt {retry_count}/{max_retries})")
            
            if retry_count >= max_retries:
                logger.error(f"Failed after {max_retries} attempts: {str(e)}")
                raise
            
            time.sleep(wait_time)

def get_or_create_user(username=None):
    """Get or create a user."""
    def _get_or_create_user():
        session = Session()
        try:
            if username:
                user = session.query(User).filter(User.username == username).first()
                if user:
                    # Create a copy of user attributes before closing session
                    user_id = user.id
                    user_username = user.username
                    user_created_at = user.created_at
                    session.close()
                    
                    # Create a new User instance with copied attributes
                    return User(id=user_id, username=user_username, created_at=user_created_at)
            
            # Create a new user
            user = User(username=username)
            session.add(user)
            session.commit()
            
            # Create a copy of user attributes before closing session
            user_id = user.id
            user_username = user.username
            user_created_at = user.created_at
            session.close()
            
            # Create a new User instance with copied attributes
            return User(id=user_id, username=user_username, created_at=user_created_at)
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()
    
    return execute_with_retry(_get_or_create_user)


def get_or_create_conversation(user_id, title=None):
    """Get or create a conversation for a user."""
    def _get_or_create_conversation():
        session = Session()
        try:
            # Check if user has any conversations
            conversation = (
                session.query(Conversation)
                .filter(Conversation.user_id == user_id)
                .order_by(Conversation.created_at.desc())
                .first()
            )
            
            if not conversation:
                # Create a new conversation
                conversation = Conversation(
                    user_id=user_id,
                    title=title or "New Conversation"
                )
                session.add(conversation)
                session.commit()
                conversation_id = conversation.id
                
                # Get the newly created conversation to ensure it was created properly
                conversation = session.query(Conversation).filter(Conversation.id == conversation_id).first()
            
            # Create a copy of conversation attributes before closing session
            conversation_id = conversation.id
            conversation_user_id = conversation.user_id
            conversation_title = conversation.title
            conversation_created_at = conversation.created_at
            
            # Close the session
            session.close()
            
            # Create a new Conversation instance with copied attributes
            return Conversation(
                id=conversation_id,
                user_id=conversation_user_id,
                title=conversation_title,
                created_at=conversation_created_at
            )
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()
    
    return execute_with_retry(_get_or_create_conversation)


def add_message_to_db(conversation_id, role, content, image_data=None):
    """Add a message to the database."""
    def _add_message():
        session = Session()
        try:
            message = Message(
                conversation_id=conversation_id,
                role=role,
                content=content,
                image_data=image_data
            )
            
            session.add(message)
            session.commit()
            return message.id
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()
    
    return execute_with_retry(_add_message)


def get_conversation_messages(conversation_id, limit=100):
    """Get messages for a conversation."""
    def _get_messages():
        session = Session()
        try:
            messages = (
                session.query(Message)
                .filter(Message.conversation_id == conversation_id)
                .order_by(Message.timestamp)
                .limit(limit)
                .all()
            )
            
            # Convert to a list of dictionaries and handle binary data
            results = []
            for msg in messages:
                message_dict = {
                    'id': msg.id,
                    'role': msg.role,
                    'content': msg.content,
                    'timestamp': msg.timestamp.strftime("%H:%M")
                }
                
                if msg.image_data:
                    message_dict['image_data'] = msg.image_data
                
                results.append(message_dict)
            
            return results
        except Exception as e:
            logger.error(f"Error getting conversation messages: {str(e)}")
            raise
        finally:
            session.close()
    
    return execute_with_retry(_get_messages)


def add_knowledge_item(content, embedding, source=None):
    """Add a knowledge item with embedding to the database."""
    def _add_knowledge_item():
        session = Session()
        try:
            # JSON serialize the embedding
            embedding_json = json.dumps(embedding)
            
            item = KnowledgeItem(
                content=content,
                embedding=embedding_json,
                source=source
            )
            
            session.add(item)
            session.commit()
            return item.id
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()
    
    return execute_with_retry(_add_knowledge_item)


def get_all_knowledge_items():
    """Get all knowledge items with embeddings."""
    def _get_items():
        session = Session()
        try:
            items = session.query(KnowledgeItem).all()
            
            # Convert to a list of dictionaries and parse embeddings
            results = []
            for item in items:
                try:
                    embedding = json.loads(item.embedding)
                    results.append({
                        'id': item.id,
                        'content': item.content,
                        'embedding': embedding,
                        'source': item.source
                    })
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing embedding JSON for item {item.id}: {str(e)}")
                    # Skip this item
                    continue
            
            return results
        except Exception as e:
            logger.error(f"Error getting knowledge items: {str(e)}")
            raise
        finally:
            session.close()
    
    return execute_with_retry(_get_items)