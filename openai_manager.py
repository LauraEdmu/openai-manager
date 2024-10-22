import os
import openai
import logging
import json
import aiofiles
import asyncio

logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s') # Configure logging to log to a file
logger = logging.getLogger(__name__) # Create a logger

console_handler = logging.StreamHandler() # Create a console handler
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') # Create a formatter and set it for the console handler
console_handler.setFormatter(formatter)

logger.addHandler(console_handler) # Add the console handler to the logger

import openai
with open("key.priv", "r") as key_file:
    api_key = key_file.read().strip()

openai.api_key = api_key

async def main():
    question = str(input("Enter Question: "))

    file_path = os.path.join("chat_history", f"chat_system")
    if os.path.exists(file_path):
        async with aiofiles.open(file_path, 'r') as f:
            system_message = await f.read()
            logger.debug(f"Read system message: {system_message}")
    else:
        logger.critical(f"Missing system message at {file_path}")
        return

    # file_path = os.path.join("chat_history", f"chat_log_{interaction.guild.id}") if getting user specific in discord
    file_path = os.path.join("chat_history", f"chat_log")
    if os.path.exists(file_path):
        async with aiofiles.open(file_path, 'r') as f:
            try:
                content = await f.read()
                messages = json.loads(content)
            except json.decoder.JSONDecodeError as e:
                logger.critical(f"Json decoder error when reading chat log. Perhaps an empty file?")
                return
            messages.append({"role": "user", "content": question})
            messages.insert(0,{"role": "system", "content": system_message})
    else:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": question}
        ]
        logger.debug(f"No file path found for chat history at {file_path}, building a new one")

    logger.info("Messages have been read/built. Preparing to send completion")

    try:
        completion = await asyncio.to_thread(
            openai.chat.completions.create,
            model="gpt-4o", # o1-preview, gpt-4o, gpt-4o-mini
            max_completion_tokens=350,
            presence_penalty=0.3,
            messages=messages
        )
    except openai.NotFoundError as e:
        logger.error("openai could not find the model")
        return
    except openai.AuthenticationError as e:
        logger.error("openai could not authenticate. Key might be read in wrong?")
        return

    logger.info(f"Completion recieved with {len(completion.choices[0].message.content)} chars. That's ~{len(completion.choices[0].message.content)/4} tokens")

    messages.append({"role": "assistant", "content": completion.choices[0].message.content})
    del messages[0]
    async with aiofiles.open(file_path, 'w') as f:
        await f.write(json.dumps(messages,indent=4))
    
    print(completion.choices[0].message.content)

if __name__ == '__main__':
    asyncio.run(main())