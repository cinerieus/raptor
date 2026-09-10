export const RaptorEnvironment = async () => ({
  "shell.env": async (_input, output) => {
    output.env.RAPTOR_AGENT = "opencode"
    output.env.OPENCODE_SESSION_ID = "direct"
  },
})
