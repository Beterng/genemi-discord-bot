import os
import re
from typing import Literal, Optional

import aiohttp
import discord
import google.generativeai as genai
from discord import *
from discord.ext import commands

#import generativeai as genai
from discord.ext.commands import Context, Greedy
from dotenv import load_dotenv
from pyvpn import *

message_history = {}

load_dotenv()

GOOGLE_AI_KEY = os.getenv("GOOGLE_AI_KEY")
DISCORD_BOT_TOKEN = os.getenv("TOKEN")
MAX_HISTORY = int(os.getenv("MAX_HISTORY"))


import imp  # pylint: disable=deprecated-module
import logging
import os
import sys


class _GenerativeAIImportHook:
  """Enables the PaLM and Gemini API clients libraries to be customized upon import."""

  def find_module(self, fullname, path=None):
    if fullname != 'google.generativeai':
      return None
    self.module_info = imp.find_module(
        fullname.split('.')[-1], list(path) if path else None
    )
    return self

  def load_module(self, fullname):
    """Loads google.generativeai normally and runs pre-initialization code.

    It runs a background server that intercepts API requests and then proxies
    the requests via the browser.

    Args:
      fullname: fullname of the module

    Returns:
      A modified google.generativeai module.
    """
    previously_loaded = fullname in sys.modules
    generativeai_module = imp.load_module(fullname, *self.module_info)

    if not previously_loaded:
      try:
        import functools  # pylint:disable=g-import-not-at-top
        import json  # pylint:disable=g-import-not-at-top

        import portpicker  # pylint:disable=g-import-not-at-top
        import tornado.web  # pylint:disable=g-import-not-at-top
        from google.colab import output  # pylint:disable=g-import-not-at-top
        from google.colab.html import (
          _background_server,  # pylint:disable=g-import-not-at-top
        )

        def fetch(request):
          path = request.path
          method = request.method
          headers = json.dumps(dict(request.headers))
          body = repr(request.body.decode('utf-8')) if request.body else 'null'
          return output.eval_js("""
            (async () => {{
              // The User-Agent header causes CORS errors in Firefox.
              const headers = {headers};
              delete headers["User-Agent"];
              const response = await fetch(new URL('{path}', 'https://generativelanguage.googleapis.com'), {{
                          method: '{method}',
                          body: {body},
                          headers,
                        }});
              const json = await response.json();
              return json;
            }})()
        """.format(path=path, method=method, headers=headers, body=body))

        class _Redirector(tornado.web.RequestHandler):
          """Redirects API requests to the browser."""

          async def get(self):
            await self._handle_request()

          async def post(self):
            await self._handle_request()

          async def _handle_request(self):
            try:
              result = fetch(self.request)
              if isinstance(result, dict) and 'error' in result:
                self.set_status(int(result['error']['code']))
                self.write(result['error']['message'])
                return
              self.write(json.dumps(result))
            except Exception as e:  # pylint:disable=broad-except
              self.set_status(500)
              self.write(str(e))

        class _Proxy(_background_server._BackgroundServer):  # pylint: disable=protected-access
          """Background server that intercepts API requests and then proxies the requests via the browser."""

          def __init__(self):
            app = tornado.web.Application([
                (r'.*', _Redirector),
            ])
            super().__init__(app)

          def create(self, port):
            if self._server_thread is None:
              self.start(port=port)

        port = portpicker.pick_unused_port()

        @functools.cache
        def start():
          p = _Proxy()
          p.create(port=port)
          return p

        start()
        orig_configure = generativeai_module.configure
        generativeai_module.configure = functools.partial(
            orig_configure,
            transport='rest',
            client_options={'api_endpoint': f'http://localhost:{port}'},
        )
      except:  # pylint: disable=bare-except
        logging.exception('Error customizing google.generativeai.')
        os.environ['COLAB_GENERATIVEAI_IMPORT_HOOK_EXCEPTION'] = '1'

    return generativeai_module


def _register_hook():
    sys.meta_path = [_GenerativeAIImportHook()] + sys.meta_path


#---------------------------------------------AI Configuration-------------------------------------------------

# Configure the generative AI model
genai.configure(api_key=GOOGLE_AI_KEY)
text_generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 512,
}
image_generation_config = {
    "temperature": 0.4,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 512,
}
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
]
text_model = genai.GenerativeModel(model_name="gemini-pro", generation_config=text_generation_config, safety_settings=safety_settings)
image_model = genai.GenerativeModel(model_name="gemini-pro-vision", generation_config=image_generation_config, safety_settings=safety_settings)


#---------------------------------------------Discord Code-------------------------------------------------
# Initialize Discord bot
bot = commands.Bot(command_prefix="!", intents=discord.Intents.default())

@bot.event
async def on_ready():
    print("----------------------------------------")
    print(f'Gemini Bot Logged in as {bot.user}')
    print("----------------------------------------")

#On Message Function
@bot.event
async def on_message(message):
    # Ignore messages sent by the bot
    if message.author == bot.user or message.mention_everyone:
        return
    # Check if the bot is mentioned or the message is a DM
    if bot.user.mentioned_in(message) or isinstance(message.channel, discord.DMChannel):
        #Start Typing to seem like something happened
        cleaned_text = clean_discord_message(message.content)

        async with message.channel.typing():
            # Check for image attachments
            if message.attachments:
                print("New Image Message FROM:" + str(message.author.id) + ": " + cleaned_text)
                #Currently no chat history for images
                for attachment in message.attachments:
                    #these are the only image extentions it currently accepts
                    if any(attachment.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']):
                        await message.add_reaction('🎨')

                        async with aiohttp.ClientSession() as session:
                            async with session.get(attachment.url) as resp:
                                if resp.status != 200:
                                    await message.channel.send('Unable to download the image.')
                                    return
                                image_data = await resp.read()
                                response_text = await generate_response_with_image_and_text(image_data, cleaned_text)
                                #Split the Message so discord does not get upset
                                await split_and_send_messages(message, response_text, 1700)
                                return
            #Not an Image do text response
            else:
                print("New Message FROM:" + str(message.author.id) + ": " + cleaned_text)
                #Check for Keyword Reset
                if "RESET" in cleaned_text:
                    #End back message
                    if message.author.id in message_history:
                        del message_history[message.author.id]
                    await message.channel.send("🤖 History Reset for user: " + str(message.author.name))
                    return
                await message.add_reaction('💬')

                #Check if history is disabled just send response
                if(MAX_HISTORY == 0):
                    response_text = await generate_response_with_text(cleaned_text)
                    #add AI response to history
                    await split_and_send_messages(message, response_text, 1700)
                    return
                #Add users question to history
                update_message_history(message.author.id,cleaned_text)
                response_text = await generate_response_with_text(get_formatted_message_history(message.author.id))
                #add AI response to history
                update_message_history(message.author.id,response_text)
                #Split the Message so discord does not get upset
                await split_and_send_messages(message, response_text, 1700)

#---------------------------------------------AI Generation History-------------------------------------------------

async def generate_response_with_text(message_text):
    prompt_parts = [message_text]
    print("Got textPrompt: " + message_text)
    response = text_model.generate_content(prompt_parts)
    if(response._error):
        return "❌" +  str(response._error)
    return response.text

async def generate_response_with_image_and_text(image_data, text):
    image_parts = [{"mime_type": "image/jpeg", "data": image_data}]
    prompt_parts = [image_parts[0], f"\n{text if text else 'What is this a picture of?'}"]
    response = image_model.generate_content(prompt_parts)
    if(response._error):
        return "❌" +  str(response._error)
    return response.text

#---------------------------------------------Message History-------------------------------------------------
def update_message_history(user_id, text):
    # Check if user_id already exists in the dictionary
    if user_id in message_history:
        # Append the new message to the user's message list
        message_history[user_id].append(text)
        # If there are more than 12 messages, remove the oldest one
        if len(message_history[user_id]) > MAX_HISTORY:
            message_history[user_id].pop(0)
    else:
        # If the user_id does not exist, create a new entry with the message
        message_history[user_id] = [text]

def get_formatted_message_history(user_id):
    """
    Function to return the message history for a given user_id with two line breaks between each message.
    """
    if user_id in message_history:
        # Join the messages with two line breaks
        return '\n\n'.join(message_history[user_id])
    else:
        return "No messages found for this user."

#---------------------------------------------Sending Messages-------------------------------------------------
async def split_and_send_messages(message_system, text, max_length):

    # Split the string into parts
    messages = []
    for i in range(0, len(text), max_length):
        sub_message = text[i:i+max_length]
        messages.append(sub_message)

    # Send each part as a separate message
    for string in messages:
        await message_system.channel.send(string)

def clean_discord_message(input_string):
    # Create a regular expression pattern to match text between < and >
    bracket_pattern = re.compile(r'<[^>]+>')
    # Replace text between brackets with an empty string
    cleaned_content = bracket_pattern.sub('', input_string)
    return cleaned_content
#
#       Code Bloacj Comment
#
# Flash Command   ----------
@bot.tree.command()

async def help(interaction: discord.Interaction):
    """Help""" #Description when viewing / commands
    await interaction.response.send_message("""
###What kind of help do you need?
###Here are some categories of assistance I can provide:
- Information: I can provide information on a wide range of topics, such as history, science, technology, and current events.
- Definitions: I can define words and phrases.
- Calculations: I can perform mathematical calculations.
- Language translation: I can translate text between multiple languages.
- Summarization: I can summarize articles and documents.
- Code generation: I can generate code in various programming languages.
- Creative writing: I can assist with creative writing tasks, such as generating story ideas or writing poems.
- Recommendations: I can provide recommendations for books, movies, music, and other products or services.
- Troubleshooting: I can help troubleshoot technical issues.
Please provide more details about the specific assistance you require, and I will do my best to help.
""")
guild = discord.Object(id='1169765046713339935')

# Get Guild ID from right clicking on server icon
# Must have devloper mode on discord on setting>Advance>Developer Mode
#More info on tree can be found on discord.py Git Repo

@bot.command()
@commands.guild_only()
@commands.is_owner()
async def sync(

  ctx: Context, guilds: Greedy[discord.Object], spec: Optional[Literal["~", "*", "^"]] = None) -> None:

    if not guilds:
        if spec == "~":
            synced = await ctx.bot.tree.sync(guild=ctx.guild)
        elif spec == "*":
            ctx.bot.tree.copy_global_to(guild=ctx.guild)
            synced = await ctx.bot.tree.sync(guild=ctx.guild)
        elif spec == "^":
            ctx.bot.tree.clear_commands(guild=ctx.guild)
            await ctx.bot.tree.sync(guild=ctx.guild)
            synced = []
        else:
            synced = await ctx.bot.tree.sync()
        await ctx.send(f"Synced {len(synced)} commands {'globally' if spec is None else 'to the current guild.'}")
        return
    ret = 0

    for guild in guilds:
        try:
            await ctx.bot.tree.sync(guild=guild)
        except discord.HTTPException:
            pass
        else:
            ret += 1
    await ctx.send(f"Synced the tree to {ret}/{len(guilds)}.")
# #
#       Code Bloacj Comment
#
#---------------------------------------------Run Bot-------------------------------------------------
bot.run(DISCORD_BOT_TOKEN)

