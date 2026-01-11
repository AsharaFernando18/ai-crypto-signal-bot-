#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test emoji sending to Telegram
"""
import os
import sys
sys.path.insert(0, 'src')

from telegram_notifier import get_notifier

# Test message with emojis
test_message = """
🤖 <b>EMOJI TEST</b> 🤖
━━━━━━━━━━━━━━

<b>📱 Testing Emojis:</b>

✅ Checkmark
❌ Cross
🎯 Target
🔥 Fire
📊 Chart
💰 Money
🚀 Rocket
⚡ Lightning

<i>If you see emojis, it works!</i>
"""

notifier = get_notifier()
if notifier.is_configured():
    result = notifier.send_message(test_message.strip())
    if result:
        print("✅ Test message sent successfully!")
    else:
        print("❌ Failed to send message")
else:
    print("❌ Telegram not configured")
