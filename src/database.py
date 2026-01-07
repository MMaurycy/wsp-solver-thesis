from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# w Dockerze zmienimy na: postgresql://user:pass@db:5432/wsp_db
SQLALCHEMY_DATABASE_URL = "sqlite:///./wsp_test.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class Event(Base):
    __tablename__ = "events"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)

class GuestGroup(Base):
    __tablename__ = "guest_groups"
    id = Column(Integer, primary_key=True, index=True)
    event_id = Column(Integer, ForeignKey("events.id"))
    name = Column(String)
    size = Column(Integer, default=1)

class Relationship(Base):
    __tablename__ = "relationships"
    id = Column(Integer, primary_key=True, index=True)
    event_id = Column(Integer, ForeignKey("events.id"))
    source_id = Column(Integer, ForeignKey("guest_groups.id"))
    target_id = Column(Integer, ForeignKey("guest_groups.id"))
    weight = Column(Integer)
    is_conflict = Column(Boolean, default=False)

def init_db():
    Base.metadata.create_all(bind=engine)
