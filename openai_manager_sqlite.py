import os
import openai
import logging
import aiofiles
import asyncio
from pathlib import Path
import subprocess
import httpcore
import aiosqlite
from colorama import Fore, Style, init

class OpenaiManager:
	def __init__(self):
		self.db_path = os.path.join("chat_history", "chat_history.db")
		
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
		async with aiosqlite.connect(self.db_path) as db:
			await db.execute('''
				CREATE TABLE IF NOT EXISTS history (
					id INTEGER PRIMARY KEY AUTOINCREMENT,
					role TEXT NOT NULL,
					content TEXT NOT NULL
				)
			''')
			await db.commit()

	async def load_key(self, keypath: str = "openai.priv") -> bool:
		if not os.path.exists(keypath):
			self.logger.error("Key not found")
			return False

		async with aiofiles.open(keypath, "r") as key_file:
			api_key = await key_file.read()

		openai.api_key = api_key
		self.logger.debug("Key read and assigned to OpenAI object")
		return True

	async def get_system_message(self, smsg_path: str ="system") -> bool:
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
			return ""
		except Exception as e:
			self.logger.critical(f"Unknown error occured: {e}")
			# raise e
			return ""

		if completion.choices[0].message.content == None:
			self.logger.warning("Completion returned None")
			return ""			

		if completion.choices[0].message.content == "":
			self.logger.warning("Completion returned empty message")
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
			return ""

		if save_path == "":
			save_path = "speech.wav"

		speech_file_path = Path(__file__).parent / save_path
		response = await asyncio.to_thread(
			openai.audio.speech.create,
			model=model,
			voice=voice, # type: ignore
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

		return str(speech_path)

	async def transcribe(self, file_path: str, srt: bool = False) -> str:
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

		# return transcription.text if not srt else transcription
		return transcription
	
	async def translate(self, file_path: str, srt: bool = False) -> str:
		if not os.path.exists(file_path):
			self.logger.error("Path does not exist for file to transcribe")
			return ""
		
		audio_file = open(file_path, "rb")
		translation = await asyncio.to_thread(
			openai.audio.translations.create,
			model="whisper-1", 
			file=audio_file,
			response_format="text" if not srt else "srt"
		)

		return translation.text if not srt else translation # type: ignore

async def quick_trans(manager, translate: bool = False):
	filename = str(input("Enter the filename: "))
	is_srt = str(input("Is the file an SRT file? (y/N): ")).lower()

	if is_srt not in ["y", "n", ""]:
		print(f"{Fore.RED}{Style.BRIGHT}Invalid input{Style.RESET_ALL}")
		return
	
	if is_srt == "y":
		is_srt = True
	else:
		is_srt = False

	if not os.path.exists(filename):
		print(f"{Fore.RED}{Style.BRIGHT}File does not exist{Style.RESET_ALL}")
		return
	
	if os.path.splitext(filename)[1] not in [".mp3", ".mp4", ".mpeg", ".mpga", ".m4a",  ".wav", ".webm"]:
		print(f"{Fore.RED}{Style.BRIGHT}File is not a valid audio file{Style.RESET_ALL}")
		return
	
	if os.path.getsize(filename)/1024/1024 > 25:
		print(f"{Fore.RED}{Style.BRIGHT}File is too large (>25MB){Style.RESET_ALL}")
		return
	
	text = await manager.transcribe(filename, srt=is_srt) if not translate else await manager.translate(filename, srt=is_srt)

	filename = filename.split(".")[0] + ".srt" if is_srt else filename.split(".")[0] + ".txt"

	out_path = os.path.join("transcriptions", filename)
	os.makedirs("transcriptions", exist_ok=True)

	async with aiofiles.open(out_path, "w") as f:
		await f.write(text)
	
	print(f"{Fore.GREEN}{Style.BRIGHT}Transcription saved to {out_path}{Style.RESET_ALL}")

async def quick_speak(manager):
	text = str(input("Enter the text to speak: "))
	voice = str(input("Enter the voice to use (shimmer, dave, etc.): "))
	filename = str(input("Enter the filename (default: speech.wav): "))
	
	if filename == "":
		filename = "speech.wav"

	speech_path = await manager.speak(text, voice=voice, save_path=filename)

	print(f"{Fore.GREEN}{Style.BRIGHT}Speech saved to {speech_path}{Style.RESET_ALL}")

async def quick_chat(manager):
	msg = str(input("Enter the message to chat: "))
	model = str(input("Enter the model to use (default: gpt-4o): "))
	max_completion_tokens = int(input("Enter the max completion tokens (default: -1): "))
	presence_penalty = float(input("Enter the presence penalty (default: -1.0): "))
	amnesia = str(input("Forget history? (y/N): ")).lower()
	history_suffix = str(input("Enter the history suffix (default: \"\"): "))
	smsg = str(input("Enter the system message (default: \"\"): "))

	if amnesia == "y":
		amnesia = True
	else:
		amnesia = False

	if history_suffix == "":
		history_suffix = None

	if smsg == "":
		smsg = None

	response = await manager.chat(msg, model=model, max_completion_tokens=max_completion_tokens, presence_penalty=presence_penalty, amnesia=amnesia, history_suffix=history_suffix, smsg=smsg)

	print(f"{Fore.GREEN}{Style.BRIGHT}Response: {response}{Style.RESET_ALL}")

async def chat_loop(manager):
	print(f"{Fore.CYAN}{Style.BRIGHT}Enter message to chat (q to quit){Style.RESET_ALL}")
	
	while True:
		msg = str(input("Msg: "))
		if msg == "q":
			break

		response = await manager.chat(msg, model="gpt-4o", smsg="Please be helpful and answer the question.")

		print(f"Response: {response}")
	
	print(f"{Fore.CYAN}{Style.BRIGHT}Exiting chat loop{Style.RESET_ALL}")

async def main():
	init() # Initialise colorama for Windows

	print(f"{Fore.CYAN}{Style.BRIGHT}Entering Main for OpenaiManager{Style.RESET_ALL}")
	
	manager = OpenaiManager()
	manager.logger.setLevel(logging.WARNING)
	
	if not await manager.load_key():
		return	

	# await quick_trans(manager, translate=False)

	await manager.get_history()
	await manager.get_system_message()

	input_msg = "Can you give me the lyrics to doja cat's say so"

	response = await manager.chat(input_msg, model="gpt-4o", smsg="Please be helpful and answer the question.")

	print(f"Response: {response}")


	input("Press Enter to exit...")


if __name__ == '__main__':
	asyncio.run(main())