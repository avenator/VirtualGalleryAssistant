import asyncio
import logging
import sys
from copy import deepcopy
from io import BytesIO

import cv2
from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart
from aiogram.types import FSInputFile, Message, BufferedInputFile
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.utils.media_group import MediaGroupBuilder
from torchvision import utils

from FaceSwap.face_detection import select_face, select_all_faces
from FaceSwap.face_swap import face_swap
from e4e_projection import projection as e4e_projection
from model import *
from util import *

# dir_path = os.path.dirname(args.out)
# if not os.path.isdir(dir_path):
#    os.makedirs(dir_path)


logging.getLogger().setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO)

args = Args()
args.correct_color = True
args.warp_2d = True
bot = Bot(token=BOT_TOKEN)

user_buffer = {}

device = 'cpu'
n_sample = 5
seed = 3000
latent_dim = 512

generator_marilyn = Generator(1024, latent_dim, 8, 2).to(device)
generator_gogh = Generator(1024, latent_dim, 8, 2).to(device)
generator_beethoven = Generator(1024, latent_dim, 8, 2).to(device)
generator_mona_lisa = Generator(1024, latent_dim, 8, 2).to(device)

generator_marilyn.load_state_dict(torch.load('my_models/generator_monro.pt', map_location=torch.device('cpu')))
generator_gogh.load_state_dict(torch.load('my_models/generator_gogh.pt', map_location=torch.device('cpu')))
generator_mona_lisa.load_state_dict(torch.load('my_models/generator_mona_lisa.pt', map_location=torch.device('cpu')))
generator_beethoven.load_state_dict(torch.load('my_models/generator_beethoven.pt', map_location=torch.device('cpu')))

generator_marilyn.eval()
generator_gogh.eval()
generator_beethoven.eval()
generator_mona_lisa.eval()

dp = Dispatcher()

monro = 'pics/monro.jpg'
beethoven = 'pics/Beethoven.jpg'
mona_lisa = 'pics/Mona-Lisa.jpg'
gogh = 'pics/Gogh.jpg'

monro_img = cv2.imread('pics/monro.jpg')
beethoven_img = cv2.imread('pics/Beethoven.jpg')
mona_img = cv2.imread('pics/Mona-Lisa.jpg')
gogh_img = cv2.imread('pics/Gogh.jpg')


@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    """
    This handler receives messages with `/start` command
    """
    await message.answer(f"Привет! Я бот - экскурсовод, "
                         f"я позволяю погрузиться в работы нашей галереи чуть глубже")

    album_builder = MediaGroupBuilder(
        caption=""
    )
    album_builder.add(
        type="photo",
        media=FSInputFile(monro)

    )
    album_builder.add(
        type="photo",
        media=FSInputFile(gogh)

    )
    album_builder.add(
        type="photo",
        media=FSInputFile(mona_lisa)

    )
    album_builder.add(
        type="photo",
        media=FSInputFile(beethoven)

    )

    await message.answer_media_group(
        media=album_builder.build()
    )

    builder = InlineKeyboardBuilder()
    builder.add(types.InlineKeyboardButton(
        text="1",
        callback_data="1")
    )
    builder.add(types.InlineKeyboardButton(
        text="2",
        callback_data="2")
    )
    builder.add(types.InlineKeyboardButton(
        text="3",
        callback_data="3")
    )
    builder.add(types.InlineKeyboardButton(
        text="4",
        callback_data="4")
    )

    await message.answer(
        text="Выберите одну из картин:",
        reply_markup=builder.as_markup()
    )


@dp.callback_query(lambda c: c.data == "1")
async def send_random_value(callback: types.CallbackQuery):
    user_buffer[callback.from_user.id] = 1
    await callback.message.answer(
        text=f"Вы выбрали: Энди Уорхол, Мэрилин Монро. Теперь, пришлите нам свое селфи!",
    )


@dp.callback_query(lambda c: c.data == "2")
async def send_random_value(callback: types.CallbackQuery):
    user_buffer[callback.from_user.id] = 2
    await callback.message.answer(
        text=f"Вы выбрали: Винсент ван Гог, Автопортрет. Теперь, пришлите нам свое селфи!",
    )


@dp.callback_query(lambda c: c.data == "3")
async def send_random_value(callback: types.CallbackQuery):
    user_buffer[callback.from_user.id] = 3
    await callback.message.answer(
        text=f"Вы выбрали Леонардо да Винчи, Портрет госпожи Лизы дель Джокондо. Теперь, пришлите нам свое селфи!",
    )


@dp.callback_query(lambda c: c.data == "4")
async def send_random_value(callback: types.CallbackQuery):
    user_buffer[callback.from_user.id] = 4
    await callback.message.answer(
        text=f"Вы выбрали Карл Штилер, Людвиг ван Бетховен. Теперь, пришлите нам свое селфи!",
    )


@dp.message(lambda c: c.photo)
async def get_image(message):
    if message.content_type == 'photo':
        img = message.photo[-1]
    else:
        img = message.document
        if img.mime_type[:5] != 'image':
            await bot.send_message(message.chat.id,
                                   "Загрузите, пожалуйста, файл в формате изображения.")
            return
    file_info = await bot.get_file(img.file_id)
    photo = await bot.download_file(file_info.file_path)
    await bot.send_message(message.chat.id,
                           "Обрабатываю ваше фото, это займет около минуты")
    # try:
    output = await style_transfer(user_buffer[message.chat.id], photo)
    await bot.send_document(message.chat.id, BufferedInputFile(deepcopy(output.getvalue()), "result.jpeg"))
    await bot.send_photo(message.chat.id, BufferedInputFile(output.getvalue(), "result.jpeg"))
    # except RuntimeError as e:
    #    print(e)
    #    await bot.send_message(message.chat.id,
    #                          "Произошла ошибка.")


async def style_transfer(style, img_filename):
    with torch.no_grad():
        my_w = flatten_face(img_filename)
        if style == 1:
            src_img = generator_marilyn(my_w, input_is_latent=True)
            dst_img = monro_img
        if style == 2:
            src_img = generator_gogh(my_w, input_is_latent=True)
            dst_img = gogh_img
        if style == 3:
            src_img = generator_mona_lisa(my_w, input_is_latent=True)
            dst_img = mona_img
        if style == 4:
            src_img = generator_beethoven(my_w, input_is_latent=True)
            dst_img = beethoven_img
    rgb = np.rollaxis(utils.make_grid(src_img[0], normalize=True).numpy(), 0, 3)
    rgb = Image.fromarray(np.uint8(rgb * 250))
    opencv = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
    src_points, src_shape, src_face = select_face(opencv)

    # Select dst face
    dst_faceBoxes = select_all_faces(dst_img)

    output = dst_img
    for k, dst_face in dst_faceBoxes.items():
        output = face_swap(src_face, dst_face["face"], src_points,
                           dst_face["points"], dst_face["shape"],
                           output, args)
    return cv2img(output)


def cv2img(img):
    is_success, buffer = cv2.imencode(".jpg", img)
    bio = BytesIO(buffer)

    return bio


def ndarray2img(arr):
    output = np.rollaxis(utils.make_grid(arr[0], normalize=True).numpy(), 0, 3)
    output = Image.fromarray(np.uint8(output * 250))

    bio = BytesIO()
    bio.name = 'result.jpeg'
    output.save(bio, 'JPEG')
    return bio


def tensor2img(t):
    output = np.rollaxis(t.detach().numpy()[0], 0, 3)
    output = Image.fromarray(np.uint8(output * 255))

    bio = BytesIO()
    bio.name = 'result.jpeg'
    output.save(bio, 'JPEG')
    return bio


def flatten_face(filepath):
    name = strip_path_extension('tmp') + '.pt'

    # aligns and crops face
    aligned_face = align_face(filepath)

    my_w = e4e_projection(aligned_face, name, device).unsqueeze(0)

    return my_w


async def main() -> None:
    # Initialize Bot instance with a default parse mode which will be passed to all API calls
    # And the run events dispatching
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
