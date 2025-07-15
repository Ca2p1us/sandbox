import axios from "axios";

// エンドポイントとパラメータを指定してPostリクエストをだす
export function postFire(
  endpoint: string,
  info: { [key: string]: string },
  isContentTypeMultiPart?: boolean
): Promise<any> {
  return new Promise((resolve, reject) => {
    let params: FormData | URLSearchParams;
    if (isContentTypeMultiPart) {
      params = new FormData();
    } else {
      params = new URLSearchParams();
    }
    Object.keys(info).forEach((key) => {
      params.append(key, info[key]);
    });
    axios({
      method: "POST",
      url: endpoint,
      responseType: "json",
      data: params,
    })
      .then((response) => {
        resolve(response);
      })
      .catch((err) => {
        console.error("err: ", err);
        reject(err);
      });
  });
}
