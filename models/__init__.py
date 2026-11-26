"""Database models and API schemas.

Importing this package registers every ORM class on the shared `Base`
metadata, which keeps SQLAlchemy relationship resolution working regardless of
import order in routers/helpers.
"""

from models.base import (
    ApiSchema,
    Base,
    ChatbotStatus,
    CrawlStatus,
    ErrorBody,
    ErrorResponse,
    PageStatus,
)
from models.chatbot import (
    Chatbot,
    ChatbotCreate,
    ChatbotListOut,
    ChatbotOut,
    ChatbotStats,
    ChatbotUpdate,
)
from models.conversation import (
    Conversation,
    ConversationCreate,
    ConversationListOut,
    ConversationOut,
    ConversationSource,
)
from models.crawl_job import CrawlJob, CrawlJobOut, CrawlStatusOut
from models.customization import ChatbotAsset
from models.message import (
    ChatConfigOut,
    ChatRequest,
    ChatResponse,
    Message,
    MessageListOut,
    MessageOut,
    MessageRole,
    RetrievalDebug,
    SourceReference,
)
from models.user import AuthenticatedUser, Profile, ProfileOut
from models.website_page import (
    WebsitePage,
    WebsitePageDetail,
    WebsitePageListOut,
    WebsitePageSummary,
    WebsitePageUpdate,
)

__all__ = [
    "ApiSchema",
    "AuthenticatedUser",
    "Base",
    "Chatbot",
    "ChatbotAsset",
    "ChatbotCreate",
    "ChatbotListOut",
    "ChatbotOut",
    "ChatbotStats",
    "ChatbotStatus",
    "ChatbotUpdate",
    "ChatConfigOut",
    "ChatRequest",
    "ChatResponse",
    "Conversation",
    "ConversationCreate",
    "ConversationListOut",
    "ConversationOut",
    "ConversationSource",
    "CrawlJob",
    "CrawlJobOut",
    "CrawlStatus",
    "CrawlStatusOut",
    "ErrorBody",
    "ErrorResponse",
    "Message",
    "MessageListOut",
    "MessageOut",
    "MessageRole",
    "PageStatus",
    "Profile",
    "ProfileOut",
    "RetrievalDebug",
    "SourceReference",
    "WebsitePage",
    "WebsitePageDetail",
    "WebsitePageListOut",
    "WebsitePageSummary",
    "WebsitePageUpdate",
]
