#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Upload telegram_commands.py with proper UTF-8 encoding
"""
import sys

# Read local file
with open('src/telegram_commands.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Write to temp file with explicit UTF-8
with open('telegram_commands_utf8.py', 'w', encoding='utf-8', newline='\n') as f:
    f.write(content)

print("✅ File converted to UTF-8 with LF line endings")
print(f"File size: {len(content)} characters")

# Check for emojis
emoji_count = sum(1 for c in content if ord(c) > 127)
print(f"Non-ASCII characters (including emojis): {emoji_count}")
