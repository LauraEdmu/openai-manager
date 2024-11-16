import os
import openai
import logging
import json
import aiofiles
import asyncio
from pathlib import Path
import subprocess
import httpcore
import aiosqlite  # Added import

class OpenaiManager:
	def __init__(self):
		self.logger = logging.getLogger(self.__class__.__name__)
		self.logger.setLevel(logging.DEBUG)

		logdir = "logs"
		os.makedirs(logdir, exist_ok=True)
		logfile = os.path.join(logdir, "OpenaiManager.log")
		
		# Create handlers if they aren't already set up
		if not self.logger.hasHandlers():
			console_handler = logging.StreamHandler()
			console_handler.setLevel(logging.DEBUG)
			file_handler = logging.FileHandler(logfile)
			file_handler.setLevel(logging.DEBUG)
			formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
			console_handler.setFormatter(formatter)
			file_handler.setFormatter(formatter)
			self.logger.addHandler(console_handler)
			self.logger.addHandler(file_handler)

		self.logger.debug("Initialised OpenaiManager instance")

	async def init_db(self):
		# self.db_path = "chat_history.db"
		self.db_path = os.path.join("chat_history", "chat_history.db")
		async with aiosqlite.connect(self.db_path) as db:
			await db.execute('''
				CREATE TABLE IF NOT EXISTS history (
					id INTEGER PRIMARY KEY AUTOINCREMENT,
					role TEXT NOT NULL,
					content TEXT NOT NULL
				)
			''')
			await db.commit()

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

	async def get_history(self) -> bool:
		self.history = []
		async with aiosqlite.connect(self.db_path) as db:
			async with db.execute('SELECT role, content FROM history') as cursor:
				async for row in cursor:
					self.history.append({"role": row[0], "content": row[1]})
		self.logger.debug("History loaded from database")
		return True

	async def chat(self, msg: str, model: str = "gpt-4o", max_completion_tokens: int = -1, presence_penalty: float = -1.0, amnesia: bool = False, history_suffix: str = "", smsg: str = "") -> str:
		if msg == "":
			self.logger.error("Message must not be empty")
			return ""

		if smsg == "":
			system = self.smsg
		else:
			system = smsg
			self.logger.debug("Using system message override (passed into .chat() method)")

		if not amnesia:
			history = self.history
		else:
			history = []

		try:
			# Prepare the parameters with conditionally added arguments
			params = {
				"model": model,
				"messages": [{"role": "system", "content": system}] + history + [{"role": "user", "content": msg}]
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
		except httpcore.LocalProtocolError as e:
			self.logger.critical("Potential key error")
		except Exception as e:
			self.logger.critical(f"Unknown error occured: {e}")
			# raise e
			return ""


		self.logger.info(f"Completion recieved with {len(completion.choices[0].message.content)} chars. That's ~{len(completion.choices[0].message.content)/4} tokens")

		if not amnesia:
			self.history.append({"role": "user", "content": msg})
			self.history.append({"role": "assistant", "content": completion.choices[0].message.content})

			async with aiosqlite.connect(self.db_path) as db:
				await db.execute('INSERT INTO history (role, content) VALUES (?, ?)', ("user", msg))
				await db.execute('INSERT INTO history (role, content) VALUES (?, ?)', ("assistant", completion.choices[0].message.content))
				await db.commit()
				self.logger.debug("History updated in database")

		return completion.choices[0].message.content

	async def clear_history(self) -> bool:
		async with aiosqlite.connect(self.db_path) as db:
			await db.execute('DELETE FROM history')
			await db.commit()
			self.logger.debug("Cleared history in database")
			return True

	async def speak(self, msg: str, voice: str = "shimmer", model: str = "tts-1", save_path: str = "") -> str:
		if msg == "":
			self.logger.warning("Cannot speak message since message is blank")
			return False

		if save_path == "":
			save_path = "speech.wav"

		speech_file_path = Path(__file__).parent / save_path
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
			f"{save_path}_normalised.wav"
		]

		try:
			await asyncio.to_thread(subprocess.run, ffmpeg_command, check=True)
			self.logger.debug("Normalised the audio file")
			speech_path = f"{save_path}_normalised.wav"
		except subprocess.CalledProcessError as e:
			self.logger.error(f"Could not normalise audio file. ffmpeg returned error code. {e}")
			speech_path = speech_file_path

		return speech_path

	async def transcribe(self, file_path: str, srt=False) -> str:
		if not os.path.exists(file_path):
			self.logger.error("Path does not exist for file to transcribe")
			return ""
		
		audio_file = open(file_path, "rb")
		transcription = await asyncio.to_thread(
			openai.audio.transcriptions.create,
			model="whisper-1", 
			file=audio_file,
			response_format="text" if not srt else "srt"
		)

		return transcription.text if not srt else transcription

async def quick_trans(manager):
	filename = str(input("Enter the filename: "))
	is_srt = str(input("Is the file an SRT file? (y/N): ")).lower()

	if is_srt not in ["y", "n"]:
		print("Invalid input")
		return
	
	if is_srt == "y":
		is_srt = True
	else:
		is_srt = False

	if not os.path.exists(filename):
		print("File does not exist")
		return
	
	text = await manager.transcribe(filename, srt=is_srt)

	filename = filename.split(".")[0] + ".srt" if is_srt else filename.split(".")[0] + ".txt"

	out_path = os.path.join("transcriptions", filename)
	os.makedirs("transcriptions", exist_ok=True)

	async with aiofiles.open(out_path, "w") as f:
		await f.write(text)

async def main():
	manager = OpenaiManager()
	
	if not await manager.load_key():
		return	

	await quick_trans(manager)


if __name__ == '__main__':
	asyncio.run(main())