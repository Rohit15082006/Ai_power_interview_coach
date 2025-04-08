import tkinter as tk
from tkinter import ttk, messagebox
import re
import json
import os
import hashlib

class loginSystem:
    def __init__(self, master, on_successful_login):
        self.master = master
        self.on_successful_login = on_successful_login
        
        # Setup the main window
        self.master.title("Interview Coach - Login")
        self.master.geometry("500x400")
        self.master.minsize(500, 400)
        
        # Load existing user data
        self.users_file = "users.json"
        self.users = self.load_users()
        
        # Create the main frame
        self.main_frame = ttk.Frame(self.master, padding="20")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create login/register frames
        self.setup_login_frame()
        self.setup_register_frame()
        
        # Show login frame by default
        self.show_login_frame()
    
    def load_users(self):
        """Load users from file or create empty dict if file doesn't exist"""
        if os.path.exists(self.users_file):
            try:
                with open(self.users_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_users(self):
        """Save users to file"""
        with open(self.users_file, 'w') as f:
            json.dump(self.users, f)
    
    def setup_login_frame(self):
        """Create login frame with username and password fields"""
        self.login_frame = ttk.Frame(self.main_frame)
        
        # Title
        title_label = ttk.Label(
            self.login_frame, 
            text="Interview Coach Login", 
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=(0, 20))
        
        # Username
        username_frame = ttk.Frame(self.login_frame)
        username_frame.pack(fill=tk.X, pady=5)
        
        username_label = ttk.Label(username_frame, text="Username:", width=12)
        username_label.pack(side=tk.LEFT)
        
        self.login_username = ttk.Entry(username_frame, width=30)
        self.login_username.pack(side=tk.LEFT, padx=5)
        
        # Password
        password_frame = ttk.Frame(self.login_frame)
        password_frame.pack(fill=tk.X, pady=5)
        
        password_label = ttk.Label(password_frame, text="Password:", width=12)
        password_label.pack(side=tk.LEFT)
        
        self.login_password = ttk.Entry(password_frame, show="•", width=30)
        self.login_password.pack(side=tk.LEFT, padx=5)
        
        # Login button
        login_button = ttk.Button(
            self.login_frame,
            text="Login",
            command=self.login
        )
        login_button.pack(pady=20)
        
        # Register link
        register_frame = ttk.Frame(self.login_frame)
        register_frame.pack(pady=10)
        
        register_text = ttk.Label(
            register_frame,
            text="Don't have an account? "
        )
        register_text.pack(side=tk.LEFT)
        
        register_link = ttk.Label(
            register_frame,
            text="Create one",
            foreground="blue",
            cursor="hand2"
        )
        register_link.pack(side=tk.LEFT)
        register_link.bind("<Button-1>", lambda e: self.show_register_frame())
    
    def setup_register_frame(self):
        """Create registration frame with validation"""
        self.register_frame = ttk.Frame(self.main_frame)
        
        # Title
        title_label = ttk.Label(
            self.register_frame, 
            text="Create New Account", 
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=(0, 20))
        
        # Full Name
        name_frame = ttk.Frame(self.register_frame)
        name_frame.pack(fill=tk.X, pady=5)
        
        name_label = ttk.Label(name_frame, text="Full Name:", width=15)
        name_label.pack(side=tk.LEFT)
        
        self.reg_fullname = ttk.Entry(name_frame, width=30)
        self.reg_fullname.pack(side=tk.LEFT, padx=5)
        
        # Username
        username_frame = ttk.Frame(self.register_frame)
        username_frame.pack(fill=tk.X, pady=5)
        
        username_label = ttk.Label(username_frame, text="Username:", width=15)
        username_label.pack(side=tk.LEFT)
        
        self.reg_username = ttk.Entry(username_frame, width=30)
        self.reg_username.pack(side=tk.LEFT, padx=5)
        
        username_hint = ttk.Label(
            self.register_frame,
            text="Username must be at least 8 characters (letters and numbers allowed)",
            font=("Arial", 8)
        )
        username_hint.pack(anchor=tk.W, padx=(110, 0))
        
        # Email
        email_frame = ttk.Frame(self.register_frame)
        email_frame.pack(fill=tk.X, pady=5)
        
        email_label = ttk.Label(email_frame, text="Email:", width=15)
        email_label.pack(side=tk.LEFT)
        
        self.reg_email = ttk.Entry(email_frame, width=30)
        self.reg_email.pack(side=tk.LEFT, padx=5)
        
        # Password
        password_frame = ttk.Frame(self.register_frame)
        password_frame.pack(fill=tk.X, pady=5)
        
        password_label = ttk.Label(password_frame, text="Password:", width=15)
        password_label.pack(side=tk.LEFT)
        
        self.reg_password = ttk.Entry(password_frame, show="•", width=30)
        self.reg_password.pack(side=tk.LEFT, padx=5)
        
        password_hint = ttk.Label(
            self.register_frame,
            text=" Password must: ""\n"
        "- Be at least 8 characters long\n"
        "- Contain at least one uppercase letter\n"
        "- Contain at least one lowercase letter\n"
        "- Contain at least one digit\n"
        "- Contain at least one special character (!@#$%^&*-_+=?)",
            font=("Arial", 8)
        )
        password_hint.pack(anchor=tk.W, padx=(110, 0))
        
        # Confirm Password
        confirm_frame = ttk.Frame(self.register_frame)
        confirm_frame.pack(fill=tk.X, pady=5)
        
        confirm_label = ttk.Label(confirm_frame, text="Confirm Password:", width=15)
        confirm_label.pack(side=tk.LEFT)
        
        self.reg_confirm = ttk.Entry(confirm_frame, show="•", width=30)
        self.reg_confirm.pack(side=tk.LEFT, padx=5)
        
        # Register button
        register_button = ttk.Button(
            self.register_frame,
            text="Create Account",
            command=self.register
        )
        register_button.pack(pady=10)
        
        # Back to login link
        login_frame = ttk.Frame(self.register_frame)
        login_frame.pack(pady=5)
        
        login_text = ttk.Label(
            login_frame,
            text="Already have an account? "
        )
        login_text.pack(side=tk.LEFT)
        
        login_link = ttk.Label(
            login_frame,
            text="Login",
            foreground="blue",
            cursor="hand2"
        )
        login_link.pack(side=tk.LEFT)
        login_link.bind("<Button-1>", lambda e: self.show_login_frame())
    
    def show_login_frame(self):
        """Show login frame and hide register frame"""
        self.register_frame.pack_forget()
        self.login_frame.pack(fill=tk.BOTH, expand=True)
        
    def show_register_frame(self):
        """Show register frame and hide login frame"""
        self.login_frame.pack_forget()
        self.register_frame.pack(fill=tk.BOTH, expand=True)
    
    def hash_password(self, password):
        """Simple password hashing using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def validate_username(self, username):
        """Validates that username is at least 8 characters and can be alphanumeric"""
        if not username:
            return False, "Username cannot be empty"
        
        if len(username) < 8:
            return False, "Username must be at least 8 characters long"
        
        if not re.match(r'^[a-zA-Z0-9]+$', username):
            return False, "Username can only contain letters and numbers"
        
        return True, "Valid username"
    
    def validate_password(self, password):
        """Validates that password meets complex requirements"""
        if not password:
            return False, "Password cannot be empty"
        
        # Check length
        if len(password) < 8:
            return False, "Password must be at least 8 characters long"
        
        # Check for uppercase letter
        if not re.search(r'[A-Z]', password):
            return False, "Password must contain at least one uppercase letter"
        
        # Check for lowercase letter
        if not re.search(r'[a-z]', password):
            return False, "Password must contain at least one lowercase letter"
        
        # Check for digit
        if not re.search(r'[0-9]', password):
            return False, "Password must contain at least one digit"
        
        # Check for special character
        if not re.search(r'[!@#$%^&*\-_+=?]', password):
            return False, "Password must contain at least one special character (!@#$%^&*-_+=?)"
        
        return True, "Valid password"
    
    def validate_email(self, email):
        """Validates email format"""
        if not email:
            return False, "Email cannot be empty"
        
        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
            return False, "Invalid email format"
        
        return True, "Valid email"
    
    def login(self):
        """Handle login attempt"""
        username = self.login_username.get()
        password = self.login_password.get()
        
        if not username or not password:
            messagebox.showerror("Error", "Please enter both username and password")
            return
        
        # Check if username exists and password matches
        if username in self.users and self.users[username]["password"] == self.hash_password(password):
            messagebox.showinfo("Success", f"Welcome back, {username}!")
            self.master.destroy()
            self.on_successful_login()
        else:
            messagebox.showerror("Error", "Invalid username or password")
    
    def register(self):
        """Handle registration attempt with validation"""
        fullname = self.reg_fullname.get()
        username = self.reg_username.get()
        email = self.reg_email.get()
        password = self.reg_password.get()
        confirm = self.reg_confirm.get()
        
        # Validate fullname
        if not fullname:
            messagebox.showerror("Error", "Please enter your full name")
            return
        
        # Validate username
        valid_username, username_msg = self.validate_username(username)
        if not valid_username:
            messagebox.showerror("Error", username_msg)
            return
        
        # Check if username already exists
        if username in self.users:
            messagebox.showerror("Error", "Username already exists. Please choose another.")
            return
        
        # Validate email
        valid_email, email_msg = self.validate_email(email)
        if not valid_email:
            messagebox.showerror("Error", email_msg)
            return
        
        # Validate password
        valid_password, password_msg = self.validate_password(password)
        if not valid_password:
            messagebox.showerror("Error", password_msg)
            return
        
        # Validate password confirmation
        if password != confirm:
            messagebox.showerror("Error", "Passwords do not match")
            return
        
        # Create new user
        self.users[username] = {
            "fullname": fullname,
            "email": email,
            "password": self.hash_password(password)
        }
        
        # Save to file
        self.save_users()
        
        messagebox.showinfo("Success", "Account created successfully. You can now login.")
        self.show_login_frame()
        
        # Clear registration fields
        self.reg_fullname.delete(0, tk.END)
        self.reg_username.delete(0, tk.END)
        self.reg_email.delete(0, tk.END)
        self.reg_password.delete(0, tk.END)
        self.reg_confirm.delete(0, tk.END)
        
        # Pre-fill login username
        self.login_username.delete(0, tk.END)
        self.login_username.insert(0, username)
        self.login_password.focus()

def launch_main_application():
    """Start the main Interview Coach application"""
    root = tk.Tk()
    from InterviewCoach import InterviewCoachGUI  # Import from your existing code
    app = InterviewCoachGUI(root)
    root.mainloop()

def main():
    # Create a login window first
    login_root = tk.Tk()
    login_system = loginSystem(login_root, launch_main_application)
    login_root.mainloop()

if __name__ == "__main__":
    main()