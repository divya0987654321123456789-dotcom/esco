"""
ESCO Intelligence Suite - Comprehensive Security Module
========================================================
This module provides complete security features including:
- Password hashing and verification (bcrypt)
- CSRF Protection
- Role-Based Access Control (RBAC)
- Rate Limiting
- Security Headers
- Input Validation & Sanitization
- Audit Logging
- Session Security
- API Token Management
"""

import os
import re
import sys
import hmac
import hashlib
import secrets
import functools
from datetime import datetime, timedelta
from collections import defaultdict
import threading
import bleach
from flask import (
    request, session, redirect, url_for, flash, 
    jsonify, render_template, g, current_app, abort
)
from flask_login import current_user

# =============================================================================
# PASSWORD SECURITY
# =============================================================================

try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False
    print("WARNING: bcrypt not installed. Using fallback hashing (less secure)")

def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt (preferred) or SHA-256 fallback.
    Returns a string that can be stored in the database.
    """
    if not password:
        raise ValueError("Password cannot be empty")
    
    if BCRYPT_AVAILABLE:
        # Use bcrypt with a work factor of 12 (secure default)
        salt = bcrypt.gensalt(rounds=12)
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    else:
        # Fallback: SHA-256 with salt (less secure but better than plaintext)
        salt = secrets.token_hex(32)
        salted = salt + password
        hashed = hashlib.sha256(salted.encode('utf-8')).hexdigest()
        return f"sha256${salt}${hashed}"

def verify_password(password: str, hashed: str) -> bool:
    """
    Verify a password against a hash.
    Supports both bcrypt hashes and legacy plaintext (for migration).
    """
    if not password or not hashed:
        return False
    
    # Check if it's a bcrypt hash
    if hashed.startswith('$2'):
        if BCRYPT_AVAILABLE:
            try:
                return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
            except Exception:
                return False
        return False
    
    # Check if it's our SHA-256 fallback format
    if hashed.startswith('sha256$'):
        parts = hashed.split('$')
        if len(parts) == 3:
            salt = parts[1]
            stored_hash = parts[2]
            salted = salt + password
            computed_hash = hashlib.sha256(salted.encode('utf-8')).hexdigest()
            return hmac.compare_digest(computed_hash, stored_hash)
        return False
    
    # Legacy: plaintext comparison (for migration purposes)
    # IMPORTANT: This should be removed after all passwords are migrated
    return hmac.compare_digest(password, hashed)

def is_password_hashed(password_field: str) -> bool:
    """Check if a password field contains a hashed password."""
    if not password_field:
        return False
    return password_field.startswith('$2') or password_field.startswith('sha256$')

def password_strength_check(password: str) -> dict:
    """
    Check password strength and return detailed feedback.
    Returns a dict with 'valid' boolean and 'errors' list.
    """
    errors = []
    
    if len(password) < 8:
        errors.append("Password must be at least 8 characters long")
    
    if not re.search(r'[A-Z]', password):
        errors.append("Password must contain at least one uppercase letter")
    
    if not re.search(r'[a-z]', password):
        errors.append("Password must contain at least one lowercase letter")
    
    if not re.search(r'\d', password):
        errors.append("Password must contain at least one number")
    
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        errors.append("Password must contain at least one special character")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'strength': calculate_password_strength(password)
    }

def calculate_password_strength(password: str) -> str:
    """Calculate password strength as weak/medium/strong."""
    score = 0
    if len(password) >= 8: score += 1
    if len(password) >= 12: score += 1
    if re.search(r'[A-Z]', password): score += 1
    if re.search(r'[a-z]', password): score += 1
    if re.search(r'\d', password): score += 1
    if re.search(r'[!@#$%^&*(),.?":{}|<>]', password): score += 1
    
    if score <= 2:
        return 'weak'
    elif score <= 4:
        return 'medium'
    return 'strong'

# =============================================================================
# CSRF PROTECTION
# =============================================================================

def generate_csrf_token() -> str:
    """Generate a CSRF token and store it in the session."""
    if '_csrf_token' not in session:
        session['_csrf_token'] = secrets.token_hex(32)
    return session['_csrf_token']

def validate_csrf_token(token: str = None) -> bool:
    """Validate the CSRF token from a form submission."""
    if token is None:
        token = request.form.get('_csrf_token') or request.headers.get('X-CSRF-Token')
    
    session_token = session.get('_csrf_token')
    if not session_token or not token:
        return False
    
    return hmac.compare_digest(session_token, token)

def csrf_protect(f):
    """Decorator to enforce CSRF protection on POST/PUT/DELETE requests."""
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if request.method in ['POST', 'PUT', 'DELETE', 'PATCH']:
            if not validate_csrf_token():
                if request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({
                        'error': 'csrf_invalid',
                        'message': 'Invalid or missing CSRF token. Please refresh the page and try again.'
                    }), 403
                flash('Security token expired. Please try again.', 'error')
                return redirect(request.referrer or url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# =============================================================================
# ROLE-BASED ACCESS CONTROL (RBAC)
# =============================================================================

# Define role hierarchy and permissions
ROLE_HIERARCHY = {
    'itadmin': 100,      # Highest level - full system access
    'it_admin': 100,     # Legacy alias
    'admin': 100,        # Legacy alias
    'topleveladmin': 90,   # Executive access - dashboards, analytics
    'top_level_admin': 90, # Legacy alias
    'supervisor': 80,     # Project oversight
    'manager': 70,        # Department management (scoped by company+department context)
    'project manager': 70,
    'teamlead': 60,       # Team management (scoped by company+department context)
    'team_lead': 60,      # Legacy alias
    # Backward-compatible aliases (older deployments stored dept names in role)
    'business_dev': 60,
    'business dev': 60,
    'design': 60,         # now treated as marketing team-lead in newer model
    'marketing': 60,
    'operations': 60,
    'site_engineer': 60,
    'site engineer': 60,
    'site_manager': 60,
    'site manager': 60,
    'member': 30,         # Regular team member
    'employee': 20,       # Employee access
    'guest': 10,          # Guest/limited access
}

# Module access permissions
MODULE_PERMISSIONS = {
    'top_admin_dashboard': ['itadmin', 'topleveladmin'],
    # Allow business development to view supervisor dashboard per access request
    'supervisor_dashboard': ['itadmin', 'topleveladmin', 'supervisor', 'manager', 'teamlead', 'employee', 'member'],
    'master_dashboard': ['itadmin', 'topleveladmin', 'supervisor'],
    'user_management': ['itadmin'],
    # Database Management is powerful; allow IT Admin + Supervisors + Top Level Admin.
    # Business Managers (BDM) can also access when operating in Business department context.
    'database_management': ['itadmin', 'topleveladmin', 'supervisor', 'business_manager'],
    'bid_analyzer': ['itadmin', 'topleveladmin', 'supervisor', 'manager', 'teamlead', 'member'],
    'manager_dashboard': ['itadmin', 'topleveladmin', 'supervisor', 'manager', 'teamlead'],
    'project_manager_dashboard': ['itadmin', 'topleveladmin', 'supervisor', 'project manager'],
    'team_lead_dashboard': ['itadmin', 'topleveladmin', 'supervisor', 'manager', 'teamlead'],
    'profiling': ['itadmin', 'topleveladmin', 'supervisor'],
    'knowledge': ['itadmin', 'topleveladmin', 'supervisor'],
    'team_business': ['itadmin', 'topleveladmin', 'supervisor', 'teamlead', 'manager'],
    'team_design': ['itadmin', 'topleveladmin', 'supervisor', 'teamlead', 'manager'],
    'team_operations': ['itadmin', 'topleveladmin', 'supervisor', 'teamlead', 'manager'],
    'team_engineer': ['itadmin', 'topleveladmin', 'supervisor', 'teamlead', 'manager'],
    'employee_dashboard': ['itadmin', 'topleveladmin', 'supervisor', 'employee', 'member', 'teamlead', 'manager'],
    'assigned_tasks': ['itadmin', 'topleveladmin', 'supervisor', 'manager', 'project manager', 'teamlead', 'employee', 'member'],
    'approvals': ['itadmin', 'topleveladmin', 'supervisor', 'manager', 'project manager', 'teamlead'],
    'bid_timeline': ['itadmin', 'topleveladmin', 'supervisor', 'manager', 'project manager', 'teamlead'],
    'business_manager_dashboard': ['itadmin', 'topleveladmin', 'supervisor', 'manager', 'business_manager'],
    'logs': ['itadmin', 'topleveladmin', 'supervisor'],
    'databases': ['itadmin', 'topleveladmin', 'supervisor', 'business_manager'],
}

# Action permissions (granular control)
ACTION_PERMISSIONS = {
    'create_user': ['itadmin'],
    'delete_user': ['itadmin'],
    'edit_user': ['itadmin'],
    'view_all_users': ['itadmin'],
    'create_bid': ['itadmin', 'supervisor', 'manager', 'teamlead'],
    'delete_bid': ['itadmin', 'supervisor'],
    'assign_stage': ['itadmin', 'supervisor', 'manager', 'teamlead'],
    'export_data': ['itadmin', 'topleveladmin', 'supervisor', 'manager', 'project manager'],
    'import_data': ['itadmin'],
    'manage_employees': ['itadmin', 'supervisor', 'manager', 'teamlead'],
    'view_analytics': ['itadmin', 'topleveladmin', 'supervisor'],
    'manage_permissions': ['itadmin'],
    'access_rfp_files': ['itadmin', 'topleveladmin', 'supervisor', 'manager', 'project manager', 'teamlead', 'member'],
    'upload_files': ['itadmin', 'supervisor', 'manager', 'project manager', 'teamlead', 'member'],
    'delete_files': ['itadmin', 'supervisor'],
}

def normalize_role(role: str) -> str:
    """Normalize role string for comparison."""
    if not role:
        return 'member'
    return role.lower().strip().replace(' ', '_').replace('-', '_')

def canonical_role(role: str) -> str:
    """
    Map legacy role strings to the new simplified role set.
    New model prefers: manager, teamlead, member, supervisor, topleveladmin, itadmin.
    """
    r = normalize_role(role)
    # IT admin
    if r in {'admin', 'it_admin', 'itadmin', 'it_administrator', 'it_administrator'}:
        return 'itadmin'
    if r in {'it', 'it_admin_login'}:
        return 'itadmin'

    # Top level admin
    if r in {'top_level_admin', 'top_leveladmin', 'top_level_admin'.replace('_', ''), 'topleveladmin', 'top_level_admin'.replace('_', '')}:
        return 'topleveladmin'
    if r in {'top level admin'}:
        return 'topleveladmin'

    if r in {'supervisor'}:
        return 'supervisor'
    if r in {'project_manager', 'project manager', 'projectmanager'}:
        return 'project manager'
    if r in {'manager'}:
        return 'manager'
    # Legacy department-as-role values → teamlead
    if r in {'business_dev', 'business_development', 'business', 'bdm', 'design', 'marketing', 'operations', 'site_engineer', 'site_manager', 'engineering', 'engineer'}:
        return 'teamlead'
    if r in {'team_lead', 'team lead', 'teamlead'}:
        return 'teamlead'
    if r in {'member', 'employee'}:
        return 'member'
    return r

def get_user_role() -> str:
    """Get the current user's role in normalized form."""
    if not current_user.is_authenticated:
        return 'guest'
    
    if getattr(current_user, 'is_admin', False):
        return 'itadmin'
    
    # Prefer a context-scoped role (e.g., based on active company/department membership)
    scoped = getattr(current_user, 'context_role', None) or getattr(current_user, 'scoped_role', None)
    role = scoped or getattr(current_user, 'role', 'member')
    return canonical_role(role)

def get_role_level(role: str) -> int:
    """Get the hierarchy level for a role."""
    normalized = normalize_role(role)
    return ROLE_HIERARCHY.get(normalized, 10)

def has_permission(permission: str) -> bool:
    """Check if current user has a specific permission."""
    user_role = get_user_role()
    allowed_roles = ACTION_PERMISSIONS.get(permission, [])
    return user_role in allowed_roles or user_role in ['itadmin']

def has_module_access(module: str) -> bool:
    """Check if current user has access to a module."""
    user_role = get_user_role()

    # Special-case: allow Business Managers to access Database Management when they are
    # using the Business department context (many installs store them as role='manager').
    if module in ('database_management', 'databases'):
        if user_role in ['itadmin', 'topleveladmin', 'supervisor', 'business_manager']:
            return True
        try:
            if user_role == 'manager' and (session.get('active_department_key') or '').lower() == 'business':
                return True
        except Exception:
            pass

    allowed_roles = MODULE_PERMISSIONS.get(module, [])
    return user_role in allowed_roles or user_role in ['itadmin']

def check_module_access(module: str) -> bool:
    """Alias for has_module_access for backward compatibility."""
    return has_module_access(module)

def require_role(*allowed_roles):
    """Decorator to require specific role(s) for access."""
    def decorator(f):
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            if not current_user.is_authenticated:
                if request.is_json:
                    return jsonify({'error': 'unauthorized', 'message': 'Please log in to access this resource.'}), 401
                return redirect(url_for('login'))
            
            user_role = get_user_role()
            normalized_allowed = [normalize_role(r) for r in allowed_roles]
            
            # IT Admin has access to everything
            if user_role in ['itadmin']:
                return f(*args, **kwargs)
            
            if user_role not in normalized_allowed:
                if request.is_json:
                    return jsonify({
                        'error': 'forbidden',
                        'message': 'You do not have permission to access this resource.'
                    }), 403
                return render_template('access_denied.html',
                    module='Restricted Area',
                    message='You do not have the required permissions to access this page.',
                    user_role=user_role
                ), 403
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def require_permission(permission: str):
    """Decorator to require a specific permission."""
    def decorator(f):
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            if not current_user.is_authenticated:
                if request.is_json:
                    return jsonify({'error': 'unauthorized', 'message': 'Please log in.'}), 401
                return redirect(url_for('login'))
            
            if not has_permission(permission):
                if request.is_json:
                    return jsonify({
                        'error': 'forbidden',
                        'message': f'Permission denied: {permission}'
                    }), 403
                return render_template('access_denied.html',
                    module=permission.replace('_', ' ').title(),
                    message=f'You do not have the "{permission}" permission.',
                    user_role=get_user_role()
                ), 403
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def require_admin(f):
    """Decorator to require IT Admin access."""
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            if request.is_json:
                return jsonify({'error': 'unauthorized'}), 401
            return redirect(url_for('login'))
        
        if not getattr(current_user, 'is_admin', False):
            if request.is_json:
                return jsonify({'error': 'forbidden', 'message': 'Admin access required'}), 403
            return render_template('access_denied.html',
                module='Admin Area',
                message='This area is restricted to IT Administrators only.',
                user_role=get_user_role()
            ), 403
        
        return f(*args, **kwargs)
    return decorated_function

def require_supervisor_or_admin(f):
    """Decorator to require supervisor or admin access."""
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            if request.is_json:
                return jsonify({'error': 'unauthorized'}), 401
            return redirect(url_for('login'))
        
        user_role = get_user_role()
        if user_role not in ['it_admin', 'admin', 'supervisor', 'top_level_admin']:
            if request.is_json:
                return jsonify({'error': 'forbidden', 'message': 'Supervisor access required'}), 403
            return render_template('access_denied.html',
                module='Supervisor Area',
                message='This area is restricted to Supervisors and Administrators.',
                user_role=user_role
            ), 403
        
        return f(*args, **kwargs)
    return decorated_function

# =============================================================================
# RATE LIMITING
# =============================================================================

class RateLimiter:
    """Thread-safe rate limiter for protecting endpoints."""
    
    def __init__(self):
        self.requests = defaultdict(list)
        self.lock = threading.Lock()
    
    def is_allowed(self, key: str, max_requests: int, window_seconds: int) -> tuple:
        """
        Check if a request is allowed based on rate limit.
        Returns (allowed: bool, remaining: int, reset_time: datetime)
        """
        now = datetime.now()
        window_start = now - timedelta(seconds=window_seconds)
        
        with self.lock:
            # Clean old requests
            self.requests[key] = [
                req_time for req_time in self.requests[key]
                if req_time > window_start
            ]
            
            current_count = len(self.requests[key])
            remaining = max(0, max_requests - current_count - 1)
            reset_time = now + timedelta(seconds=window_seconds)
            
            if current_count >= max_requests:
                return False, 0, reset_time
            
            self.requests[key].append(now)
            return True, remaining, reset_time
    
    def reset(self, key: str):
        """Reset rate limit for a key."""
        with self.lock:
            self.requests[key] = []

# Global rate limiter instance
rate_limiter = RateLimiter()

def rate_limit(max_requests: int = 60, window_seconds: int = 60, key_func=None):
    """
    Decorator to apply rate limiting to routes.
    Default: 60 requests per minute per IP.
    """
    def decorator(f):
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            # Generate rate limit key
            if key_func:
                key = key_func()
            else:
                # Default: use IP address and endpoint
                ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
                if ip:
                    ip = ip.split(',')[0].strip()
                key = f"rate:{ip}:{request.endpoint}"
            
            allowed, remaining, reset_time = rate_limiter.is_allowed(
                key, max_requests, window_seconds
            )
            
            # Add rate limit headers
            @after_this_request
            def add_rate_limit_headers(response):
                response.headers['X-RateLimit-Limit'] = str(max_requests)
                response.headers['X-RateLimit-Remaining'] = str(remaining)
                response.headers['X-RateLimit-Reset'] = reset_time.isoformat()
                return response
            
            if not allowed:
                if request.is_json:
                    return jsonify({
                        'error': 'rate_limit_exceeded',
                        'message': 'Too many requests. Please try again later.',
                        'retry_after': window_seconds
                    }), 429
                return render_template('access_denied.html',
                    module='Rate Limited',
                    message=f'Too many requests. Please wait {window_seconds} seconds before trying again.',
                    user_role=get_user_role() if current_user.is_authenticated else 'guest'
                ), 429
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def login_rate_limit(f):
    """Special rate limit for login endpoints - stricter limits."""
    return rate_limit(max_requests=5, window_seconds=300)(f)  # 5 attempts per 5 minutes

# Helper for after_this_request in rate limiting
def after_this_request(f):
    if not hasattr(g, 'after_request_callbacks'):
        g.after_request_callbacks = []
    g.after_request_callbacks.append(f)
    return f

# =============================================================================
# SECURITY HEADERS
# =============================================================================

def add_security_headers(response):
    """Add security headers to all responses."""
    # Prevent clickjacking
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    
    # Prevent MIME type sniffing
    response.headers['X-Content-Type-Options'] = 'nosniff'
    
    # Enable XSS protection
    response.headers['X-XSS-Protection'] = '1; mode=block'
    
    # Referrer policy
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    
    # Permissions policy (formerly Feature-Policy)
    response.headers['Permissions-Policy'] = 'geolocation=(), microphone=(), camera=()'
    
    # Content Security Policy (adjust as needed for your CDNs)
    csp = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval' "
        "https://cdn.tailwindcss.com "
        "https://cdnjs.cloudflare.com "
        "https://cdn.jsdelivr.net "
        "https://unpkg.com "
        "https://cdn.socket.io; "
        "script-src-elem 'self' 'unsafe-inline' 'unsafe-eval' "
        "https://cdn.tailwindcss.com "
        "https://cdnjs.cloudflare.com "
        "https://cdn.jsdelivr.net "
        "https://unpkg.com "
        "https://cdn.socket.io; "
        "style-src 'self' 'unsafe-inline' "
        "https://cdn.tailwindcss.com "
        "https://cdnjs.cloudflare.com "
        "https://fonts.googleapis.com; "
        "font-src 'self' "
        "https://fonts.gstatic.com "
        "https://cdnjs.cloudflare.com; "
        "img-src 'self' data: blob: https:; "
        "connect-src 'self' ws: wss: https://cdn.jsdelivr.net https://cdnjs.cloudflare.com https://cdn.tailwindcss.com https://unpkg.com; "
        "frame-ancestors 'self';"
    )
    response.headers['Content-Security-Policy'] = csp
    
    # Strict Transport Security (enable in production with HTTPS)
    # response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    
    return response

# =============================================================================
# INPUT VALIDATION & SANITIZATION
# =============================================================================

# Allowed HTML tags for rich text (very restrictive)
ALLOWED_TAGS = ['b', 'i', 'u', 'em', 'strong', 'p', 'br', 'ul', 'ol', 'li', 'a']
ALLOWED_ATTRIBUTES = {
    'a': ['href', 'title'],
}

def sanitize_html(text: str) -> str:
    """Sanitize HTML input to prevent XSS attacks."""
    if not text:
        return ''
    return bleach.clean(
        text,
        tags=ALLOWED_TAGS,
        attributes=ALLOWED_ATTRIBUTES,
        strip=True
    )

def sanitize_input(text: str) -> str:
    """Sanitize plain text input - strip all HTML."""
    if not text:
        return ''
    return bleach.clean(text, tags=[], strip=True).strip()

def validate_email(email: str) -> bool:
    """Validate email format."""
    if not email:
        return False
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_filename(filename: str) -> bool:
    """Validate filename to prevent path traversal."""
    if not filename:
        return False
    # Disallow path traversal characters
    dangerous_patterns = ['..', '/', '\\', '\x00']
    for pattern in dangerous_patterns:
        if pattern in filename:
            return False
    return True

def secure_filename(filename: str) -> str:
    """Create a secure version of a filename."""
    if not filename:
        return 'unnamed'
    # Remove path components
    filename = os.path.basename(filename)
    # Remove dangerous characters
    filename = re.sub(r'[^\w\s\-\.]', '', filename)
    # Replace spaces with underscores
    filename = filename.replace(' ', '_')
    # Limit length
    if len(filename) > 200:
        name, ext = os.path.splitext(filename)
        filename = name[:200-len(ext)] + ext
    return filename or 'unnamed'

# =============================================================================
# AUDIT LOGGING
# =============================================================================

class SecurityAuditLog:
    """Security-focused audit logging."""
    
    @staticmethod
    def log_login_attempt(email: str, success: bool, ip: str, user_agent: str = None):
        """Log a login attempt."""
        status = 'SUCCESS' if success else 'FAILED'
        message = f"LOGIN_{status}: email={email}, ip={ip}"
        if user_agent:
            message += f", ua={user_agent[:100]}"
        SecurityAuditLog._write_log('AUTH', message)
    
    @staticmethod
    def log_permission_denied(user_email: str, resource: str, action: str):
        """Log a permission denied event."""
        message = f"PERMISSION_DENIED: user={user_email}, resource={resource}, action={action}"
        SecurityAuditLog._write_log('ACCESS', message)
    
    @staticmethod
    def log_rate_limit(ip: str, endpoint: str):
        """Log a rate limit exceeded event."""
        message = f"RATE_LIMIT_EXCEEDED: ip={ip}, endpoint={endpoint}"
        SecurityAuditLog._write_log('SECURITY', message)
    
    @staticmethod
    def log_csrf_failure(ip: str, endpoint: str):
        """Log a CSRF validation failure."""
        message = f"CSRF_FAILURE: ip={ip}, endpoint={endpoint}"
        SecurityAuditLog._write_log('SECURITY', message)
    
    @staticmethod
    def log_suspicious_activity(description: str, ip: str = None, user_email: str = None):
        """Log suspicious activity."""
        message = f"SUSPICIOUS: {description}"
        if ip:
            message += f", ip={ip}"
        if user_email:
            message += f", user={user_email}"
        SecurityAuditLog._write_log('SECURITY', message)
    
    @staticmethod
    def log_admin_action(admin_email: str, action: str, target: str = None, details: str = None):
        """Log an admin action."""
        message = f"ADMIN_ACTION: admin={admin_email}, action={action}"
        if target:
            message += f", target={target}"
        if details:
            message += f", details={details}"
        SecurityAuditLog._write_log('ADMIN', message)
    
    @staticmethod
    def log_data_access(user_email: str, data_type: str, action: str, record_id: str = None):
        """Log data access events."""
        message = f"DATA_ACCESS: user={user_email}, type={data_type}, action={action}"
        if record_id:
            message += f", id={record_id}"
        SecurityAuditLog._write_log('DATA', message)
    
    @staticmethod
    def _write_log(category: str, message: str):
        """Write log entry to database and/or file."""
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] [{category}] {message}"
        
        # Print to console (for development)
        print(log_entry)
        
        # Write to file
        try:
            log_dir = os.path.join(os.getcwd(), 'logs')
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f'security_{datetime.now().strftime("%Y%m%d")}.log')
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry + '\n')
        except Exception as e:
            print(f"Failed to write security log: {e}")

# Global audit log instance
audit_log = SecurityAuditLog()

# =============================================================================
# SESSION SECURITY
# =============================================================================

def configure_session_security(app):
    """Configure secure session settings for the Flask app."""
    # Session configuration
    app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production with HTTPS
    app.config['SESSION_COOKIE_HTTPONLY'] = True  # Prevent JavaScript access
    app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # CSRF protection
    app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=8)  # Session timeout
    app.config['SESSION_REFRESH_EACH_REQUEST'] = True  # Extend session on activity
    
    # Additional security
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload
    
    # Generate a strong secret key if not set
    if not app.config.get('SECRET_KEY') or app.config['SECRET_KEY'] == 'a_very_secret_key_for_sessions':
        app.config['SECRET_KEY'] = secrets.token_hex(32)
        print("WARNING: Using auto-generated SECRET_KEY. Set a permanent key in production.")

def regenerate_session():
    """Regenerate session ID to prevent session fixation attacks."""
    # Store important session data
    csrf_token = session.get('_csrf_token')
    
    # Preserve Flask-Login session keys
    flask_login_data = {
        '_user_id': session.get('_user_id'),
        '_id': session.get('_id'),
        '_fresh': session.get('_fresh'),
        '_remember': session.get('_remember'),
    }
    
    user_data = {
        'employee_id': session.get('employee_id'),
        'employee_name': session.get('employee_name'),
        'employee_email': session.get('employee_email'),
        'employee_department': session.get('employee_department'),
    }
    
    # Clear and regenerate
    session.clear()
    session.modified = True
    
    # Restore important data
    if csrf_token:
        session['_csrf_token'] = csrf_token
    
    # Restore Flask-Login data
    for key, value in flask_login_data.items():
        if value is not None:
            session[key] = value
    
    for key, value in user_data.items():
        if value:
            session[key] = value

# =============================================================================
# API TOKEN SECURITY
# =============================================================================

def generate_api_token(user_id: int, expires_in: int = 86400) -> str:
    """Generate a secure API token for a user."""
    payload = f"{user_id}:{datetime.now().timestamp()}:{secrets.token_hex(16)}"
    signature = hmac.new(
        current_app.config['SECRET_KEY'].encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()
    return f"{payload}:{signature}"

def validate_api_token(token: str) -> tuple:
    """
    Validate an API token.
    Returns (valid: bool, user_id: int or None)
    """
    if not token:
        return False, None
    
    try:
        parts = token.rsplit(':', 1)
        if len(parts) != 2:
            return False, None
        
        payload, signature = parts
        expected_signature = hmac.new(
            current_app.config['SECRET_KEY'].encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()
        
        if not hmac.compare_digest(signature, expected_signature):
            return False, None
        
        payload_parts = payload.split(':')
        if len(payload_parts) >= 2:
            user_id = int(payload_parts[0])
            timestamp = float(payload_parts[1])
            
            # Check if token is expired (default 24 hours)
            if datetime.now().timestamp() - timestamp > 86400:
                return False, None
            
            return True, user_id
    except Exception:
        pass
    
    return False, None

def require_api_token(f):
    """Decorator to require a valid API token."""
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('X-API-Token') or request.args.get('api_token')
        
        if not token:
            return jsonify({
                'error': 'missing_token',
                'message': 'API token is required'
            }), 401
        
        valid, user_id = validate_api_token(token)
        if not valid:
            return jsonify({
                'error': 'invalid_token',
                'message': 'Invalid or expired API token'
            }), 401
        
        g.api_user_id = user_id
        return f(*args, **kwargs)
    return decorated_function

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_client_ip() -> str:
    """Get the client's IP address, handling proxies."""
    if request.environ.get('HTTP_X_FORWARDED_FOR'):
        return request.environ['HTTP_X_FORWARDED_FOR'].split(',')[0].strip()
    return request.remote_addr or 'unknown'

def is_safe_url(target: str) -> bool:
    """Check if a URL is safe for redirects (same origin)."""
    if not target:
        return False
    from urllib.parse import urlparse, urljoin
    ref_url = urlparse(request.host_url)
    test_url = urlparse(urljoin(request.host_url, target))
    return test_url.scheme in ('http', 'https') and ref_url.netloc == test_url.netloc

def get_safe_redirect(target: str, fallback: str = None) -> str:
    """Get a safe redirect URL or fallback."""
    if target and is_safe_url(target):
        return target
    return fallback or url_for('dashboard')

# =============================================================================
# FLASK APP INTEGRATION
# =============================================================================

def init_security(app):
    """Initialize security features for the Flask app."""
    # Configure session security
    configure_session_security(app)
    
    # Add CSRF token generator to templates
    @app.context_processor
    def security_context():
        return {
            'csrf_token': generate_csrf_token,
            'has_permission': has_permission,
            'has_module_access': has_module_access,
            'get_user_role': get_user_role,
        }
    
    # Add security headers to all responses
    @app.after_request
    def apply_security_headers(response):
        response = add_security_headers(response)
        
        # Execute rate limit callbacks
        if hasattr(g, 'after_request_callbacks'):
            for callback in g.after_request_callbacks:
                response = callback(response)
        
        return response
    
    # Log failed login attempts
    @app.errorhandler(401)
    def handle_401(error):
        ip = get_client_ip()
        audit_log.log_suspicious_activity('Unauthorized access attempt', ip=ip)
        if request.is_json:
            return jsonify({'error': 'unauthorized', 'message': 'Please log in to continue.'}), 401
        return redirect(url_for('login'))
    
    @app.errorhandler(403)
    def handle_403(error):
        ip = get_client_ip()
        user_email = getattr(current_user, 'email', 'anonymous') if current_user.is_authenticated else 'anonymous'
        audit_log.log_permission_denied(user_email, request.path, request.method)
        if request.is_json:
            return jsonify({'error': 'forbidden', 'message': 'You do not have permission to access this resource.'}), 403
        return render_template('access_denied.html',
            module='Restricted',
            message='You do not have permission to access this resource.',
            user_role=get_user_role() if current_user.is_authenticated else 'guest'
        ), 403
    
    try:
        print("✓ Security module initialized")
    except UnicodeEncodeError:
        print("Security module initialized")
    return app


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Password functions
    'hash_password',
    'verify_password',
    'is_password_hashed',
    'password_strength_check',
    
    # CSRF
    'generate_csrf_token',
    'validate_csrf_token',
    'csrf_protect',
    
    # RBAC
    'ROLE_HIERARCHY',
    'MODULE_PERMISSIONS',
    'ACTION_PERMISSIONS',
    'normalize_role',
    'get_user_role',
    'get_role_level',
    'has_permission',
    'has_module_access',
    'check_module_access',
    'require_role',
    'require_permission',
    'require_admin',
    'require_supervisor_or_admin',
    
    # Rate limiting
    'rate_limiter',
    'rate_limit',
    'login_rate_limit',
    
    # Security headers
    'add_security_headers',
    
    # Input validation
    'sanitize_html',
    'sanitize_input',
    'validate_email',
    'validate_filename',
    'secure_filename',
    
    # Audit logging
    'audit_log',
    'SecurityAuditLog',
    
    # Session security
    'configure_session_security',
    'regenerate_session',
    
    # API security
    'generate_api_token',
    'validate_api_token',
    'require_api_token',
    
    # Utilities
    'get_client_ip',
    'is_safe_url',
    'get_safe_redirect',
    
    # App initialization
    'init_security',
]
