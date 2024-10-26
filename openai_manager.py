import os
import openai
import logging
import json
import aiofiles
import asyncio
from pathlib import Path
import subprocess

class OpenaiManager:
	def __init__(self):
		self.logger = logging.getLogger(self.__class__.__name__)
		self.logger.setLevel(logging.DEBUG)
		
		# Create handlers if they aren't already set up
		if not self.logger.hasHandlers():
			console_handler = logging.StreamHandler()
			console_handler.setLevel(logging.DEBUG)
			file_handler = logging.FileHandler('OpenaiManager.log')
			file_handler.setLevel(logging.DEBUG)
			formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
			console_handler.setFormatter(formatter)
			file_handler.setFormatter(formatter)
			self.logger.addHandler(console_handler)
			self.logger.addHandler(file_handler)

		self.logger.debug("Initialised OpenaiManager instance")

	async def load_key(self, keypath="openai.priv") -> bool:
		if not os.path.exists(keypath):
			self.logger.error("Key not found")
			return False

		async with aiofiles.open(keypath, "r") as key_file:
			api_key = await key_file.read()

		openai.api_key = api_key
		self.logger.debug("Key read and assigned to OpenAI object")
		return True

	async def get_system_message(self, smsg_path="system") -> bool:
		smsg_path = os.path.join("chat_history", smsg_path)
		if not os.path.exists(smsg_path):
			self.logger.error("Could not find system message")
			return False

		async with aiofiles.open(smsg_path, 'r') as f:
			self.smsg = await f.read()

		self.logger.debug("System message read")
		return True

	async def get_history(self, suffix="") -> bool:
		file_path = os.path.join(f"chat_history", f"chat_log{suffix}")
		if os.path.exists(file_path):
			async with aiofiles.open(file_path, 'r') as f:
				try:
					content = await f.read()
					self.history = json.loads(content)
				except json.decoder.JSONDecodeError as e:
					self.logger.error(f"Json decoder error when reading chat log. Perhaps an empty file?")
					return False
				self.history.insert(0,{"role": "system", "content": self.smsg})
		else:
			self.logger.debug(f"No file path found for chat history at {file_path}. Initialised empty history")
			self.history = []

		return True

	async def chat(self, msg: str, model: str = "gpt-4o", max_completion_tokens: int = -1, presence_penalty: float = -1.0, amnesia: bool = False, history_suffix: str = "", smgs: str = "") -> str:
		if msg == "":
			self.logger.error("Message must not be empty")
			return ""

		if smsg == "":
			system = self.smsg
		else:
			system = smsg
			logger.debug("Using system message override (passed into .chat() method)")

		try:
			# Prepare the parameters with conditionally added arguments
			params = {
				"model": model,
				"messages": [{"role": "system", "content": system}] + self.history + [{"role": "user", "content": msg}]
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
			self.logger.error("openai could not find the model")
			return ""
		except openai.AuthenticationError as e:
			self.logger.error("openai could not authenticate. Key might be read in wrong?")
			return ""
		except Exception as e:
			self.logger.critical(f"Unknown error occured: {e}")
			# raise e
			return ""


		self.logger.info(f"Completion recieved with {len(completion.choices[0].message.content)} chars. That's ~{len(completion.choices[0].message.content)/4} tokens")

		if not amnesia:
			self.history.append({"role": "user", "content": msg})
			self.history.append({"role": "assistant", "content": completion.choices[0].message.content})

			try:
				file_path = os.path.join(f"chat_history", f"chat_log{history_suffix}")
				async with aiofiles.open(file_path, 'w') as f:
					await f.write(json.dumps(self.history, indent=4))

				self.logger.debug(f"Wrote history to {file_path}")
			except Exception as e:
				self.logger.error(f"Could not write history to {file_path}. {e}")

		return completion.choices[0].message.content

	def clear_history(self, suffix="") -> bool:
		file_path = os.path.join(f"chat_history", f"chat_log{suffix}")
		if os.path.exists(file_path):
			os.remove(file_path)
			self.logger.debug(f"Removed file {file_path}")
			return True
		else:
			self.logger.warning(f"History was asked to clear, but no history was found")
			return False

	async def speak(self, msg: str, voice: str = "shimmer", model: str = "tts-1") -> bool:
		if msg == "":
			self.logger.warning("Cannot speak message since message is blank")
			return False

		speech_file_path = Path(__file__).parent / "speech.wav"
		response = await asyncio.to_thread(
			openai.audio.speech.create,
			model=model,
			voice=voice,
			input=msg,
			response_format="wav"
		)
		response.stream_to_file(speech_file_path)

		ffmpeg_command = [
			"ffmpeg",
			"-y", 
			"-i", speech_file_path,
			"-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
			"-v", "quiet",
			"speech_normalised.wav"
		]

		try:
			await asyncio.to_thread(subprocess.run, ffmpeg_command, check=True)
			self.logger.debug("Normalised the audio file")
			speech_path = "speech_normalised.wav"
		except subprocess.CalledProcessError as e:
			self.logger.error(f"Could not normalise audio file. ffmpeg returned error code. {e}")
			speech_path = speech_file_path

		return True

	async def transcribe(self, file_path: str) -> str:
		if not os.path.exists(file_path):
			self.logger.error("Path does not exist for file to transcribe")
			return ""
		
		audio_file = open(file_path, "rb")
		transcription = await asyncio.to_thread(
			openai.audio.transcriptions.create,
			model="whisper-1", 
			file=audio_file
			# response_format="vtt"
		)

		return transcription.text

async def main():
	manager = OpenaiManager()

	import elevenlabs_manager
	from playsound import playsound
	eleven = elevenlabs_manager.ElevenlabsManager()

	if not await eleven.load_key():
		return
	
	if not await manager.load_key():
		return	

	if not await manager.get_system_message():
		return

	if not await manager.get_history():
		return

	response = await manager.chat("Can you write me a haiku about yourself")

	print(response)

	audio_path = await eleven.speak(response)
	playsound(audio_path)

	

if __name__ == '__main__':
	asyncio.run(main())