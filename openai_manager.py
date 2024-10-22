import os
import openai
import logging
import json
import aiofiles
import asyncio
import openai

logging.basicConfig(filename='Openai_Manager.log', level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s') # Configure logging to log to a file
logger = logging.getLogger(__name__) # Create a logger

console_handler = logging.StreamHandler() # Create a console handler
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') # Create a formatter and set it for the console handler
console_handler.setFormatter(formatter)

logger.addHandler(console_handler) # Add the console handler to the logger

class OpenaiManager:
	def __init__(self):
		logger.debug("Initialised OpenaiManager instance")

	async def load_key(self, keypath="openai.priv") -> bool:
		if not os.path.exists(keypath):
			logger.error("Key not found")
			return False

		async with aiofiles.open(keypath, "r") as key_file:
			api_key = await key_file.read()

		openai.api_key = api_key
		logger.debug("Key read and assigned to OpenAI object")
		return True

	async def get_system_message(self, smsg_path="system") -> bool:
		smsg_path = os.path.join("chat_history", smsg_path)
		if not os.path.exists(smsg_path):
			logger.error("Could not find system message")
			return False

		async with aiofiles.open(smsg_path, 'r') as f:
			self.smsg = await f.read()

		logger.debug("System message read")
		return True

	async def get_history(self, suffix="") -> bool:
		file_path = os.path.join(f"chat_history", f"chat_log{suffix}")
		if os.path.exists(file_path):
			async with aiofiles.open(file_path, 'r') as f:
				try:
					content = await f.read()
					self.history = json.loads(content)
				except json.decoder.JSONDecodeError as e:
					logger.error(f"Json decoder error when reading chat log. Perhaps an empty file?")
					return False
				self.history.insert(0,{"role": "system", "content": self.smsg})
		else:
			logger.debug(f"No file path found for chat history at {file_path}. Initialised empty history")
			self.history = []

		return True

	async def chat(self, msg: str, model: str = "gpt-4o", max_completion_tokens: int = -1, presence_penalty: float = -1.0, amnesia: bool = False, history_suffix: str = "") -> str:
		if msg == "":
			logger.error("Message must not be empty")
			return ""

		try:
			# Prepare the parameters with conditionally added arguments
			params = {
				"model": model,
				"messages": [{"role": "system", "content": self.smsg}] + self.history + [{"role": "user", "content": msg}]
			}
			if max_completion_tokens != -1:
				params["max_completion_tokens"] = max_completion_tokens
			if presence_penalty != -1.0:
				params["presence_penalty"] = presence_penalty

			# Call the API with the prepared parameters
			completion = await asyncio.to_thread(
				openai.chat.completions.create,
				**params
			)
		except openai.NotFoundError as e:
			logger.error("openai could not find the model")
			return ""
		except openai.AuthenticationError as e:
			logger.error("openai could not authenticate. Key might be read in wrong?")
			return ""
		except Exception as e:
			logger.critical(f"Unknown error occured: {e}")
			return ""

		logger.info(f"Completion recieved with {len(completion.choices[0].message.content)} chars. That's ~{len(completion.choices[0].message.content)/4} tokens")

		if not amnesia:
			self.history.append({"role": "user", "content": msg})
			self.history.append({"role": "assistant", "content": completion.choices[0].message.content})

			try:
				file_path = os.path.join(f"chat_history", f"chat_log{history_suffix}")
				async with aiofiles.open(file_path, 'w') as f:
					await f.write(json.dumps(self.history, indent=4))

				logger.debug(f"Wrote history to {file_path}")
			except Exception as e:
				logger.error(f"Could not write history to {file_path}. {e}")

		return completion.choices[0].message.content

	def clear_history(self, suffix="") -> bool:
		file_path = os.path.join(f"chat_history", f"chat_log{suffix}")
		if os.path.exists(file_path):
			os.remove(file_path)
			logger.debug(f"Removed file {file_path}")
			return True
		else:
			logger.warning(f"History was asked to clear, but no history was found")
			return False



async def main():
	manager = OpenaiManager()
	
	if not await manager.load_key():
		return	

	if not await manager.get_system_message():
		return

	if not await manager.get_history():
		return

	response = await manager.chat("Do you like talking about weather?")

	print(response)

	# manager.clear_history()

if __name__ == '__main__':
	asyncio.run(main())